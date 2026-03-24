# Projekt 5 — Stick Animation (Diffusion)

Generacja animacji szkieletu (walk / jump) modelem dyfuzyjnym, sterowana
klasą ruchu. Wyjście to tensor `[48, 15, 3]` (klatki × stawy × XYZ), który
jest następnie wizualizowany jako GIF.

## Inspiracja i czym różni się ta implementacja

Punktem wyjścia była publiczna realizacja:
[FilipLangiewicz / ComputerVisionAndAIImageProcessing — `project5-stick-animation`](https://github.com/FilipLangiewicz/ComputerVisionAndAIImageProcessing/tree/main/project5-stick-animation).
Sample dla klasy `walk` w tamtym repo łapią ogólny rytm chodu, ale postać
"oddycha" — kości się rozciągają i kurczą między klatkami, a stopy
ślizgają się po podłodze. Główna przyczyna to to, że model pracuje na
spłaszczonym wektorze `[T, 45]` w przestrzeni świata, bez żadnej wiedzy o
strukturze szkieletu, długościach kości ani o tym że pelvis jest korzeniem
hierarchii.

Tu robimy to inaczej, opierając się o materiał z wykładów W12 "Wstęp do
animacji" i W12 "Animacja postaci – interpolacja":

| Aspekt | Repo Langiewicza | Ten projekt |
|---|---|---|
| Reprezentacja | `[T, 15·3]` w świecie, centrowanie pelvis-em | dekompozycja root motion (xy_vel + z + yaw sin/cos) + 14 lokalnych stawów po cancelacji yaw |
| Normalizacja | jedna skalarna mean/std | per-cecha mean/std + per-sekwencja `bone_scale` |
| Architektura | flat transformer encoder, 1 token per klatka | spatio-temporal DiT, **15 tokenów per klatka** (1 token = 1 staw) z osobnymi embeddingami i biasem atencji z grafu szkieletu |
| Inspiracja architekturą | brak | PoseFormerV2 (W12, slajd 60-62) — **dodatkowe tokeny DCT** dla niskoczęstotliwościowych wzorców ruchu |
| Schedule szumu | linear betas | **cosine** (Nichol & Dhariwal) |
| Parametryzacja | ε-prediction | **v-prediction** (Salimans & Ho) |
| Sampling | full DDPM 1000 kroków | **DDIM** 50 kroków (~20× szybciej) z opcjonalną stochastycznością |
| Conditioning | suma embeddingów | **AdaLN-Zero** (DiT) — startuje jako identyczność |
| Loss | MSE szumu + velocity MSE | MSE (v lub eps) + **bone-length consistency** + **smoothness** (acceleration) + **foot-skating** + velocity |
| Augmentacja | rotate + mirror | rotate + mirror + time-warp + speed |
| Post-processing | brak | **snap długości kości** po sampling (rekonstrukcja FK od pelvis w dół) |

Każda zmiana ma uzasadnienie z wykładu lub z literatury — patrz "Decyzje
projektowe" niżej.

## Decyzje projektowe (mapowanie na materiał wykładów)

* **Hierarchiczna reprezentacja, pelvis-root** (W12 "Wstęp do animacji",
  slajd 23: *Szkielet postaci jest drzewiastą strukturą hierarchiczną.
  Dla postaci ludzkich — korzeń drzewa — miednica*). Lokalne pozycje są
  liczone względem pelvis-a, dodatkowo w układzie po cancelacji yaw, żeby
  model nie musiał uczyć się różnych orientacji świata oddzielnie.

* **Długości kości jako twardy invariant** (W12 "Wstęp do animacji",
  slajd 22: *łańcuchy składają się ze sztywnych obiektów połączonych
  stawami tzw. kości*). `L_bone` karze za niespójne długości kości w
  predykowanym `x_0`, a `snap_bone_lengths` na koniec rzutuje wynik do
  prawidłowych długości w kolejności BFS od pelvis-a (forward kinematics
  bez rotacji jako ich aproksymacja).

* **Smoothness loss (acceleration)** (W12 "Animacja postaci", slajd 55:
  `E_smooth` — *Stabilność w czasie*). Druga pochodna pozycji w czasie
  karze jitter, który DDPM-y produkują przy małych t.

* **Foot skating** (W12 "Wstęp do animacji", slajd 12: *Interpolacja jest
  zwykle generowana, czasami poprawiana przez animatora* — chodzi o to,
  że w realnej animacji stopa na podłodze nie ślizga się). Soft contact
  gate na minimalnej wysokości kostki + kara za jej prędkość poziomą.

* **DCT branch + Fuzja czasowo-częstotliwościowa** (W12 "Animacja
  postaci", slajd 60-62, PoseFormerV2: *wykorzystanie DCT do przekształcenia
  sekwencji do dziedziny częstotliwości. Niskoczęstotliwościowe
  współczynniki przechowują główne wzorce ruchu*). 8 niskoczęstotliwościowych
  współczynników DCT trafia jako dodatkowe tokeny do warstwy temporalnej.

* **Skeleton attention bias.** Atencja spatial dostaje dodatkowy logit
  `-dystans_w_grafie(j_a, j_b)`, więc bliższe stawy w drzewie kinematycznym
  domyślnie silniej się "widzą".

* **MPJPE** (W12 "Animacja postaci", slajd 57) jest jedną z wymaganych
  miar — wraz z FMD i Var liczone po reprezentacji root-relative (żeby
  nie dominowała globalna translacja).

## Reprezentacja danych

Każda sekwencja `[T, 15, 3]` w przestrzeni świata jest dekomponowana na:

| Pole | Rozmiar | Znaczenie |
|---|---|---|
| `root_xy_vel` | `[T, 2]` | prędkość pelvis-a w XY (jeden krok do tyłu w rekonstrukcji = `cumsum`) |
| `root_z` | `[T, 1]` | wysokość pelvis-a |
| `root_yaw` | `[T, 2]` | `(sin yaw, cos yaw)` — kierunek twarzy |
| `local` | `[T, 14, 3]` | pozostałe stawy w lokalnym układzie pelvis-a (po wyzerowaniu yaw) |

Łącznie `F = 47` cech na klatkę. Wszystko skalowane przez per-sekwencyjny
`bone_scale` (średnia długość kości w pozie spoczynkowej), żeby cechy
miały porównywalne skale niezależnie od wzrostu podmiotu.

## Architektura

Spatio-temporal DiT:

```
input [B, T, 47]
    │ split: root[B,T,5]  local[B,T,14,3]
    ▼
per-joint linear projection + joint identity embedding
    [B, T, 15, D]
    │ × n_layers
    ├── spatial DiT block (per-frame, atencja po stawach + skeleton bias)
    └── temporal DiT block (per-staw, atencja po [klatki ∥ tokeny DCT])
    ▼
LayerNorm
split heads:
  root head → [B, T, 5]
  local head → [B, T, 14, 3]
    ▼
output v_pred [B, T, 47]
```

* Każdy blok DiT używa **AdaLN-Zero** (DiT, Peebles & Xie) — projekcja
  conditioning-u na 6×D produkuje (shift, scale, gate)×2.
* Inicjalizacja gate=0 sprawia, że na starcie model = identyczność.
* DCT-II współczynniki tworzą `n_dct_tokens` (domyślnie 8) tokenów na
  staw, dołączanych do osi czasowej, z osobnym embeddingiem "rodzaju".

## Diffusion

* `Diffusion(timesteps=1000, schedule="cosine", parametrization="v")`
* Strata: MSE(v_target, v_pred) + `0.1·L_vel + 0.5·L_bone + 0.05·L_smooth + 0.2·L_foot`
* Sampling: DDIM (`n_steps=50`, η=0) + CFG (`guidance=3.0`)
* Po samplingu opcjonalny `snap_bone_lengths` (BFS od pelvis-a, zachowuje
  kierunek kości, ustawia długość na wartość rest).

## Instalacja

```bash
uv sync                     # synchronizuje .venv z pyproject.toml + uv.lock
source .venv/bin/activate   # opcjonalnie, jeśli wolisz aktywować shell
# albo wszystkie polecenia poniżej puszczaj przez `uv run …`
```

(Względem poprzednich projektów dochodzą zależności: `bvhio`,
`scikit-learn`, `einops`, `pandas`.)

## Przygotowanie danych

1. Klonuj zbiór CMU MoCap (z PDF zadania) i wybierz BVH-ki pasujące do
   klas `walk` / `jump` (patrz `cmu-mocap-index-spreadsheet.xls`):

   ```bash
   git clone https://github.com/una-dinosauria/cmu-mocap.git
   # skopiuj wybrane .bvh do data/raw/walk i data/raw/jump
   ```

2. Zbuduj sklasyfikowane, ucanonicalizowane tensory:

   ```bash
   uv run python -m src.stick_animation.prepare_data \
       --raw-dir data/raw \
       --out-dir data/stickanim
   ```

   Powstanie `data/stickanim/{train,test}.npz`, `norm_stats.npy` i
   `rest_bones.npy`. Tensor `sequences` ma kształt `[N, 48, 47]`.

## Trening

```bash
uv run python train_stickanim.py --data-dir data/stickanim --epochs 400
```

Trening domyślnie używa EMA (decay 0.999) i co 25 epok wypluwa GIF-y oraz
siatkę pozy końcowej dla obu klas pod `output/stickanim/samples/eXXX/`.
Pełna ewaluacja FMD/MPJPE/Var na zbiorze testowym uruchamia się
automatycznie po treningu i zapisuje raport JSON w
`output/stickanim/metrics_final.json`.

## Sampling z gotowego checkpointa

```bash
uv run python train_stickanim.py --data-dir data/stickanim \
    --skip-train --ckpt output/stickanim/final.pt
```

## Eksperymenty dodatkowe

`experiments_stickanim.py` puszcza zbiór ablacji i sweepów (długo działa —
domyślnie 200 epok na trening i 12 eksperymentów):

```bash
uv run python experiments_stickanim.py \
    --data-dir data/stickanim \
    --out-dir output/stickanim_experiments \
    --epochs 200
```

Wynik zostaje wyplutiony jako `output/stickanim_experiments/ablations.csv`
z jednym wierszem na (eksperyment × klasa ruchu). Eksperymenty:

| Eksperyment | Czego dotyczy |
|---|---|
| `A_schedule_cosine_v` / `A_schedule_linear_v` | wpływ harmonogramu szumu |
| `B_param_eps_cosine` | parametryzacja ε vs `v` |
| `C_steps_{25,50,100,1000}` | liczba kroków DDIM przy stałym checkpoincie |
| `D_cfg_{1,2,3,5,7.5}` | wpływ siły CFG na FMD/MPJPE/Var |
| `E_loss_no_{bone,smooth,foot,geom}` | ablacja stratny geometrycznej |
| `F_arch_no_dct` | wpływ branchu DCT |
| `G_snap_{off,on}` | wpływ "snap" kości w post-processing |

## Miary (zgodnie z PDF)

| Miara | Implementacja |
|---|---|
| FMD | mean+cov cech (mean pose, std pose, mean vel, std vel) → dystans Frécheta |
| MPJPE | per-joint per-frame L2, sparowane z najbliższym sąsiadem w cechach |
| Var | średnia parowych odległości w cechach + std pozycji i prędkości |

Wszystkie liczone po root-relative reprezentacji, żeby translacja globalna
nie zaburzała wyniku.

| Ruch | FMD | MPJPE | Var (pairwise) |
|---|---|---|---|
| walk | (po treningu) | (po treningu) | (po treningu) |
| jump | (po treningu) | (po treningu) | (po treningu) |

## Struktura katalogów

```
src/stick_animation/
  __init__.py
  skeleton.py            # joint enum, hierarchia, kości, mirror, graf
  representation.py      # world ↔ canonical, FK, snap kości, DCT
  data_loader.py         # BVH → [T, 15, 3]
  prepare_data.py        # split + augmentacja + normalizacja
  dataset.py             # PyTorch Dataset
  diffusion.py           # cosine schedule + v/eps + DDIM + CFG
  losses.py              # multi-objective loss
  metrics.py             # FMD, MPJPE, Var
  sampling.py            # DDIM → world → bone snap
  visualize.py           # animate_skeleton_3d + grid_static
  models/
    spatiotemporal_dit.py

train_stickanim.py          # trening + ewaluacja
experiments_stickanim.py    # ablacje + sweepy
```
