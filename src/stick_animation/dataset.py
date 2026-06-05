"""PyTorch ``Dataset`` over the canonicalized motion features produced by ``prepare_data``."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


class MotionDataset(Dataset):
    def __init__(self, npz_path: str):
        data = np.load(npz_path)
        self.sequences = torch.from_numpy(data["sequences"]).float()        # [N, T, F]
        self.labels = torch.from_numpy(data["labels"]).long()                # [N]
        self.bone_scales = torch.from_numpy(data["bone_scales"]).float()     # [N]
        self.rest_bones = torch.from_numpy(data["rest_bones"]).float()       # [N, n_bones]

    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x0": self.sequences[idx],
            "label": self.labels[idx],
            "bone_scale": self.bone_scales[idx],
            "rest_bones": self.rest_bones[idx],
        }


def load_norm_stats(out_dir: str) -> Dict[str, np.ndarray]:
    p = Path(out_dir)
    stats = np.load(p / "norm_stats.npy")
    rest = np.load(p / "rest_bones.npy")
    return {"mean": stats[0], "std": stats[1], "rest_bones": rest}
