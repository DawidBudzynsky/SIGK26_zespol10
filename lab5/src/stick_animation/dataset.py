from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.sequences = torch.from_numpy(data["sequences"]).float()   # [N, T, 45]
        self.labels = torch.from_numpy(data["labels"]).long()           # [N]

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"x0": self.sequences[idx], "label": self.labels[idx]}


def load_norm_stats(out_dir: str) -> Dict[str, np.ndarray]:
    """Per-channel normalisation stats + rest skeleton.

    Returns mean/std as [45] arrays and rest as [n_bones]. Falls back to the
    legacy scalar ``norm_stats.npy`` format if the new .npz is absent.
    """
    npz = Path(out_dir) / "norm_stats.npz"
    if npz.exists():
        s = np.load(npz)
        return {"mean": s["mean"].astype(np.float32),
                "std": s["std"].astype(np.float32),
                "rest": s["rest"].astype(np.float32)}
    legacy = np.load(Path(out_dir) / "norm_stats.npy")
    return {"mean": np.float32(legacy[0]), "std": np.float32(legacy[1]), "rest": None}
