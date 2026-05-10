from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# parametry sceny zapisujemy relatywnie do kamery, zamiast w światowych współrzędnych. Kamera w naszym setupie jest stała (5, 5, 15), więc relatywne kodowanie zostawia siatkę z tylko geometrycznie istotnymi informacjami: kierunek + odległość obiektu/światła od kamery, a sieć nie musi się uczyć stałej transformacji widoku.
POS_SCALE = 25.0
SHIN_LO, SHIN_HI = 3.0, 20.0


def encode_params(sample: dict, camera) -> np.ndarray:
    cam   = np.asarray(camera,            dtype=np.float32)
    obj   = np.asarray(sample["obj_pos"], dtype=np.float32)
    light = np.asarray(sample["light_pos"], dtype=np.float32)
    diffuse = np.asarray(sample["diffuse"], dtype=np.float32) / 255.0
    shin = (float(sample["shininess"]) - SHIN_LO) / (SHIN_HI - SHIN_LO)

    return np.concatenate([
        (obj   - cam) / POS_SCALE,
        (light - cam) / POS_SCALE,
        diffuse,
        np.array([shin], dtype=np.float32),
    ]).astype(np.float32)


class PhongDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        with open(self.data_dir / "labels.json") as f:
            meta = json.load(f)
        self.samples = meta["samples"]
        self.camera = meta["camera"]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        feats = encode_params(s, self.camera)
        img = np.asarray(Image.open(self.data_dir / f"{s['idx']:05d}.png"))
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # CHW
        return torch.from_numpy(feats), torch.from_numpy(img)


def split_indices(n: int, test_frac: float = 0.20, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(round(n * test_frac))
    return idx[n_test:].tolist(), idx[:n_test].tolist()
