"""Build train/test ``.npz`` arrays of canonicalized motion sequences.

Input layout (expected from the user):

    raw_data/
        walk/*.bvh   (or *.npy with shape [T, 15, 3])
        jump/*.bvh

Output (under OUT_DIR):

    cached/walk/<name>.npy   raw [T, 15, 3] tensors (re-used by experiments)
    cached/jump/<name>.npy
    train.npz                sequences [N, 48, F], labels [N], bone_scales [N]
    test.npz                 same layout, no augmentation
    norm_stats.npy           per-feature mean/std for the F-dim representation

The "canonical representation" (see ``representation.py``) decomposes each
sequence into root xy velocity, root z, root yaw (sin, cos) and
yaw-cancelled local joint positions; this is what the diffusion model
learns to denoise. Augmentation is applied on the canonical feature vector
where it is cheap and exact (no geometric reconstruction).
"""
from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .data_loader import list_bvh_files, load_bvh_as_tensor, load_cached, save_cached
from .representation import MotionRep, world_to_rep
from .skeleton import MIRROR_PAIRS, N_JOINTS, Joint

LABEL_MAP = {"walk": 0, "jump": 1}
FRAMES_DEFAULT = 48


# ----------------------------------------------------------------------------
# Resampling / augmentation on canonical features
# ----------------------------------------------------------------------------


def _resample(arr: np.ndarray, n_frames: int) -> np.ndarray:
    """Linear time resample along axis 0."""
    T = arr.shape[0]
    if T == n_frames:
        return arr
    idx = np.linspace(0, T - 1, n_frames)
    left = np.floor(idx).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (idx - left).reshape(-1, *([1] * (arr.ndim - 1)))
    return (1 - alpha) * arr[left] + alpha * arr[right]


def _resample_world(world: np.ndarray, n_frames: int) -> np.ndarray:
    return _resample(world, n_frames)


def _augment_world_rotation(world: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate the whole sequence around the world Z axis."""
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return world @ R.T


def _augment_world_mirror(world: np.ndarray) -> np.ndarray:
    """Mirror the skeleton across the YZ plane (swap left/right joints + negate X)."""
    m = world.copy()
    for a, b in MIRROR_PAIRS:
        m[:, int(a), :], m[:, int(b), :] = world[:, int(b), :].copy(), world[:, int(a), :].copy()
    m[:, :, 0] *= -1.0
    return m


def _augment_world_time_warp(world: np.ndarray, max_warp: float = 0.15) -> np.ndarray:
    """Non-uniform time resample: alters tempo slightly without changing length."""
    T = world.shape[0]
    base = np.linspace(0.0, T - 1.0, T)
    rng = np.random.default_rng()
    bumps = rng.normal(0.0, max_warp, size=T - 2)
    perturbed = base.copy()
    perturbed[1:-1] += bumps
    perturbed = np.clip(perturbed, 0.0, T - 1.0)
    perturbed = np.sort(perturbed)
    left = np.floor(perturbed).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (perturbed - left).reshape(-1, 1, 1)
    return (1 - alpha) * world[left] + alpha * world[right]


def _augment_world_speed(world: np.ndarray, factor: float) -> np.ndarray:
    """Stretch or compress in time, then resample back to the original T."""
    T = world.shape[0]
    new_T = max(8, int(round(T * factor)))
    idx = np.linspace(0, T - 1, new_T)
    left = np.floor(idx).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (idx - left).reshape(-1, 1, 1)
    sped = (1 - alpha) * world[left] + alpha * world[right]
    return _resample_world(sped, T)


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------


def _load_one(path: str, n_frames: int, cache_dir: str = None) -> np.ndarray:
    """Load a single sequence as [T_n, 15, 3] world tensor, with caching."""
    if path.endswith(".npy"):
        return np.load(path)
    if cache_dir is not None:
        cached = os.path.join(cache_dir, os.path.basename(path) + ".npy")
        cached_arr = load_cached(cached)
        if cached_arr is not None:
            return cached_arr
    arr = load_bvh_as_tensor(path, frame_stride=1)
    if cache_dir is not None:
        save_cached(cached, arr)
    return arr


def _augmentations_per_class() -> dict:
    """Number of augmented copies generated per training sample, per class."""
    return {0: 7, 1: 13}  # walk has fewer subtypes; jump is rarer in CMU


def prepare(
    raw_dir: str,
    out_dir: str,
    n_frames: int = FRAMES_DEFAULT,
    seed: int = 42,
    test_size: float = 0.2,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cached")
    rng = np.random.default_rng(seed)

    all_paths: List[Tuple[str, int]] = []
    for name, label in LABEL_MAP.items():
        sub = os.path.join(raw_dir, name)
        bvhs = list_bvh_files(sub)
        npys = [os.path.join(sub, f) for f in sorted(os.listdir(sub)) if f.endswith(".npy")] if os.path.isdir(sub) else []
        for p in bvhs + npys:
            all_paths.append((p, label))
    if not all_paths:
        raise FileNotFoundError(
            f"No .bvh/.npy files under {raw_dir}/walk or {raw_dir}/jump. "
            "Place CMU MoCap trials there (see README_stickanim.md)."
        )

    print(f"Found {len(all_paths)} source sequences "
          f"({sum(1 for _,l in all_paths if l == 0)} walk, "
          f"{sum(1 for _,l in all_paths if l == 1)} jump)")

    labels_all = np.array([l for _, l in all_paths])
    train_idx, test_idx = train_test_split(
        np.arange(len(all_paths)),
        test_size=test_size,
        random_state=seed,
        stratify=labels_all,
    )

    aug_per = _augmentations_per_class()
    feature_dim = MotionRep.from_tensor(
        world_to_rep(np.zeros((n_frames, N_JOINTS, 3), dtype=np.float32)).to_tensor(),
        1.0,
    ).feature_dim

    splits = {"train": train_idx, "test": test_idx}
    for split_name, idx_arr in splits.items():
        feats: List[np.ndarray] = []
        labs: List[int] = []
        scales: List[float] = []
        rests: List[np.ndarray] = []

        for i in idx_arr:
            path, label = all_paths[i]
            try:
                world = _load_one(path, n_frames, cache_dir)
            except Exception as e:
                print(f"  skip {path}: {e}")
                continue
            if world.shape[0] < 4:
                continue
            world = _resample_world(world, n_frames)

            variants = [world]
            if split_name == "train":
                for _ in range(aug_per[label]):
                    aug = world.copy()
                    if rng.random() < 0.85:
                        aug = _augment_world_rotation(aug, rng.uniform(0, 360))
                    if rng.random() < 0.5:
                        aug = _augment_world_mirror(aug)
                    if rng.random() < 0.5:
                        aug = _augment_world_time_warp(aug, max_warp=0.2)
                    if rng.random() < 0.5:
                        aug = _augment_world_speed(aug, factor=rng.uniform(0.85, 1.15))
                    variants.append(aug)

            for w in variants:
                rep = world_to_rep(w)
                feats.append(rep.to_tensor())
                labs.append(label)
                scales.append(rep.bone_scale)
                # rest bone lengths from the (un-augmented) base motion would
                # be more accurate, but augmentation only translates / rotates
                # rigidly so this is identical.
                from .representation import rest_bone_lengths
                rests.append(rest_bone_lengths(w))

        feats_np = np.stack(feats).astype(np.float32)  # [N, T, F]
        labs_np = np.array(labs, dtype=np.int64)
        scales_np = np.array(scales, dtype=np.float32)
        rests_np = np.stack(rests).astype(np.float32)

        if split_name == "train":
            mean = feats_np.reshape(-1, feature_dim).mean(0)
            std = feats_np.reshape(-1, feature_dim).std(0) + 1e-6
            np.save(os.path.join(out_dir, "norm_stats.npy"),
                    np.stack([mean, std], axis=0))
            mean_rest = rests_np.mean(0)
            np.save(os.path.join(out_dir, "rest_bones.npy"), mean_rest)
        else:
            stats = np.load(os.path.join(out_dir, "norm_stats.npy"))
            mean, std = stats[0], stats[1]

        feats_np = (feats_np - mean) / std

        out_path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez(
            out_path,
            sequences=feats_np,
            labels=labs_np,
            bone_scales=scales_np,
            rest_bones=rests_np,
        )
        print(f"  {split_name}: {feats_np.shape}  labels={np.bincount(labs_np)}")
        print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True,
                    help="Folder with walk/ and jump/ subfolders of BVH or NPY files.")
    ap.add_argument("--out-dir", required=True, help="Where to write train.npz/test.npz.")
    ap.add_argument("--n-frames", type=int, default=FRAMES_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    args = ap.parse_args()
    prepare(args.raw_dir, args.out_dir, args.n_frames, args.seed, args.test_size)


if __name__ == "__main__":
    main()
