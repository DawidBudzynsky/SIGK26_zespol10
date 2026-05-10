"""Ewaluacja cGAN-a na zbiorze testowym (20% z 3000 = 600 obrazów).

Wylicza FLIP / LPIPS / SSIM / Hausdorff (sec. 3 PDF), zapisuje CSV per próbka,
tabelę zbiorczą i siatkę porównań (GT vs predykcja vs różnica).
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

from models import Generator
from utils.dataset import PhongDataset
from utils.metrics import all_metrics, to_uint8


def make_montage(triples, path, scale: int = 2, diff_gain: int = 4):
    """triples = [(gt, pred, label_str)]; składa 3-kolumnowy montaż."""
    rows = []
    for gt, pred, label in triples:
        diff = np.abs(gt.astype(np.int16) - pred.astype(np.int16)) * diff_gain
        diff = diff.clip(0, 255).astype(np.uint8)
        row = np.concatenate([gt, pred, diff], axis=1)
        rows.append((row, label))

    h, w = rows[0][0].shape[:2]
    canvas_h = (h + 22) * len(rows) * scale + 30
    canvas_w = w * scale + 100
    canvas = Image.new("RGB", (canvas_w, canvas_h), (28, 28, 28))
    draw = ImageDraw.Draw(canvas)
    draw.text((110, 6), "GT",                              fill=(255, 255, 255))
    draw.text((110 + (w * scale) // 3, 6), "Predykcja",    fill=(255, 255, 255))
    draw.text((110 + 2 * (w * scale) // 3, 6),
              f"|GT - Pred|  x{diff_gain}",                fill=(255, 200, 200))

    y = 30
    for row, label in rows:
        big = Image.fromarray(row).resize((w * scale, h * scale), Image.NEAREST)
        canvas.paste(big, (100, y))
        draw.text((6, y + (h * scale) // 2 - 6), label, fill=(180, 220, 255))
        y += (h * scale) + 8

    canvas.save(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data",        default="lab3/data")
    ap.add_argument("--ckpt",        default="lab3/checkpoints/cgan.pt")
    ap.add_argument("--out_dir",     default="lab3/results")
    ap.add_argument("--batch_size",  type=int, default=64)
    ap.add_argument("--n_montage",   type=int, default=8)
    args = ap.parse_args()

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    G = Generator().to(device)
    G.load_state_dict(ck["generator"])
    G.eval()
    print(f"loaded {args.ckpt}  (epoch {ck.get('epoch', '?')}, "
          f"no_gan={ck.get('no_gan', False)})")

    full = PhongDataset(args.data)
    test_ds = Subset(full, ck["test_idx"])
    print(f"test set: {len(test_ds)}")
    loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    samples = []   # (FLIP, gt_uint8, pred_uint8, idx)
    pbar = tqdm(total=len(test_ds), desc="evaluating")
    with torch.no_grad():
        offset = 0
        for params, real in loader:
            params = params.to(device)
            fake = G(params).cpu().numpy()
            real = real.numpy()
            for i in range(len(params)):
                pred = to_uint8(fake[i]); gt = to_uint8(real[i])
                m = all_metrics(pred, gt, device=str(device))
                rows.append(m)
                samples.append((m["FLIP"], gt, pred, ck["test_idx"][offset + i]))
                pbar.update(1)
            offset += len(params)
    pbar.close()

    keys = ["FLIP", "LPIPS", "SSIM", "Hausdorff"]
    means = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    stds  = {k: float(np.std([r[k] for r in rows]))  for k in keys}

    print("\nWyniki (600 testów):")
    for k in keys:
        print(f"  {k:9s} = {means[k]:.4f}  ± {stds[k]:.4f}")

    with open(out / "metrics_per_sample.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["idx"] + keys)
        w.writeheader()
        for s, r in zip(samples, rows):
            w.writerow({"idx": s[3], **r})

    with open(out / "metrics_summary.md", "w") as f:
        f.write("| Metoda | FLIP | LPIPS | SSIM | Hausdorff |\n")
        f.write("|--------|------|-------|------|-----------|\n")
        f.write(f"| neural_renderer | {means['FLIP']:.4f} "
                f"| {means['LPIPS']:.4f} | {means['SSIM']:.4f} "
                f"| {means['Hausdorff']:.4f} |\n")

    samples.sort(key=lambda s: s[0])
    n = args.n_montage
    half = n // 2
    pick_good = np.linspace(0, len(samples) // 4, half).astype(int)
    pick_bad  = np.linspace(int(len(samples) * 0.85), len(samples) - 1, n - half).astype(int)
    triples = []
    for p in list(pick_good) + list(pick_bad):
        flip_v, gt, pred, idx = samples[p]
        label = ("typowy" if p in pick_good else "trudny") + f"\nFLIP={flip_v:.3f}"
        triples.append((gt, pred, label))
    make_montage(triples, out / "qualitative_montage.png")

    print(f"wrote {out}/{{metrics_per_sample.csv, metrics_summary.md, qualitative_montage.png}}")


if __name__ == "__main__":
    main()
