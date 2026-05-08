"""Dataset reading the Phong rendered images + scene parameter labels."""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def encode_params_absolute(sample: dict, camera) -> np.ndarray:
    """Ablation encoding (spec hint #1): absolute world-frame positions.

    No camera relativisation, no spherical-coordinate features. Only the raw
    object/light xyz (scaled to [-1, 1] over the [-20, 20] sample range), the
    diffuse colour, and shininess.
    """
    obj = np.array(sample["obj_pos"], dtype=np.float32) / 20.0
    light = np.array(sample["light_pos"], dtype=np.float32) / 20.0
    diffuse = np.array(sample["diffuse"], dtype=np.float32) / 255.0
    shininess = (float(sample["shininess"]) - 3.0) / 17.0
    pad = np.zeros(15 - 3 - 3 - 3 - 1, dtype=np.float32)
    return np.concatenate([
        obj, light, diffuse, np.array([shininess], dtype=np.float32), pad
    ]).astype(np.float32)


def encode_params(sample: dict, camera) -> np.ndarray:
    """Encode a scene into a fixed feature vector.

    Following spec hint #1, position values are encoded *relative to the
    camera* (object-relative-to-camera and light-relative-to-camera) so that
    the generator only needs to learn what the camera frustum looks like
    rather than the absolute world frame. Each axis is normalised to roughly
    [-1, 1] by dividing by 30 (the max plausible distance).

    Output features (15 floats):
      [0:3]   object position relative to camera (normalised)
      [3:6]   light  position relative to camera (normalised)
      [6:9]   diffuse color in [0, 1]
      [9]     shininess scaled to [0, 1] (range [3, 20])
      [10]    sin/cos extras: distance to camera, normalised
      [11:13] sin/cos of azimuth from camera to object
      [13:15] sin/cos of elevation from camera to object
    """
    cam = np.array(camera, dtype=np.float32)
    obj = np.array(sample["obj_pos"], dtype=np.float32)
    light = np.array(sample["light_pos"], dtype=np.float32)
    diffuse = np.array(sample["diffuse"], dtype=np.float32) / 255.0
    shininess = float(sample["shininess"])

    obj_rel = (obj - cam) / 30.0
    light_rel = (light - cam) / 30.0

    dist = np.linalg.norm(obj - cam) / 30.0
    delta = obj - cam
    azimuth = math.atan2(delta[0], -delta[2])
    elevation = math.atan2(delta[1], math.hypot(delta[0], delta[2]))

    feats = np.concatenate([
        obj_rel,
        light_rel,
        diffuse,
        np.array([(shininess - 3.0) / 17.0]),
        np.array([dist]),
        np.array([math.sin(azimuth), math.cos(azimuth)]),
        np.array([math.sin(elevation), math.cos(elevation)]),
    ]).astype(np.float32)
    return feats


PARAM_DIM = 15


class PhongDataset(Dataset):
    def __init__(self, data_dir: str, indices=None, encoding: str = "relative"):
        self.data_dir = Path(data_dir)
        self.encoding = encoding
        with open(self.data_dir / "labels.json") as f:
            meta = json.load(f)
        self.samples = meta["samples"]
        self.camera = meta["camera"]
        if indices is not None:
            self.samples = [self.samples[i] for i in indices]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        if self.encoding == "absolute":
            feats = encode_params_absolute(s, self.camera)
        else:
            feats = encode_params(s, self.camera)
        img = np.array(Image.open(self.data_dir / f"{s['idx']:05d}.png"))
        if img.shape[-1] == 4:
            img = img[..., :3]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))   # CHW
        return torch.from_numpy(feats), torch.from_numpy(img)


def split_indices(n_total: int, test_frac: float = 0.2, seed: int = 0):
    rng = np.random.RandomState(seed)
    idx = np.arange(n_total)
    rng.shuffle(idx)
    n_test = int(round(n_total * test_frac))
    return idx[n_test:].tolist(), idx[:n_test].tolist()
