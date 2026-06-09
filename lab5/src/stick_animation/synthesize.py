from __future__ import annotations

from typing import List

import numpy as np

from .representation import canonical_to_world, world_to_canonical
from .skeleton import Joint


def _resample(arr: np.ndarray, n_frames: int) -> np.ndarray:
    T = arr.shape[0]
    if T == n_frames:
        return arr
    idx = np.linspace(0, T - 1, n_frames)
    left = np.floor(idx).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (idx - left).reshape(-1, *([1] * (arr.ndim - 1)))
    return (1 - alpha) * arr[left] + alpha * arr[right]


def _apex_align(world: np.ndarray, n_frames: int) -> np.ndarray:
    z = world[:, int(Joint.PELVIS), 2]
    apex = int(np.argmax(z))
    half = n_frames // 2
    apex = min(max(apex, 1), world.shape[0] - 2)
    rise = _resample(world[: apex + 1], half)
    fall = _resample(world[apex:], n_frames - half)
    return np.concatenate([rise, fall], axis=0)


def synthesize_jumps(
    jump_worlds: List[np.ndarray],
    n_synth: int,
    rest_bones: np.ndarray,
    seed: int = 0,
    n_frames: int = 48,
) -> List[np.ndarray]:
    if len(jump_worlds) < 2 or n_synth <= 0:
        return []
    rng = np.random.default_rng(seed)
    canon = [world_to_canonical(_apex_align(w, n_frames)) for w in jump_worlds]
    out: List[np.ndarray] = []
    for _ in range(n_synth):
        i, j = rng.choice(len(canon), size=2, replace=False)
        a = float(rng.uniform(0.3, 0.7))
        feat = (1.0 - a) * canon[i] + a * canon[j]
        out.append(canonical_to_world(feat, rest_bones=rest_bones))
    return out
