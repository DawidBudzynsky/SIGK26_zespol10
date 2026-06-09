from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
from sklearn.model_selection import train_test_split

from .data_loader import list_bvh_files, load_bvh_as_tensor, load_cached, save_cached
from .representation import rest_bone_lengths, world_to_canonical
from .skeleton import MIRROR_PAIRS, N_JOINTS, Joint
from .synthesize import synthesize_jumps

LABEL_MAP = {"walk": 0, "jump": 1}
FRAMES_DEFAULT = 48
FEATURE_DIM = N_JOINTS * 3


def _resample(arr: np.ndarray, n_frames: int) -> np.ndarray:
    T = arr.shape[0]
    if T == n_frames:
        return arr
    idx = np.linspace(0, T - 1, n_frames)
    left = np.floor(idx).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (idx - left).reshape(-1, *([1] * (arr.ndim - 1)))
    return (1 - alpha) * arr[left] + alpha * arr[right]


def _center(world: np.ndarray) -> np.ndarray:
    """Subtract mean pelvis position so the sequence starts near origin."""
    pelvis_mean = world[:, int(Joint.PELVIS), :].mean(axis=0, keepdims=True)
    pelvis_mean[:, 2] = 0.0
    return world - pelvis_mean[:, None, :]


def _rotate(world: np.ndarray, angle_deg: float) -> np.ndarray:
    a = np.deg2rad(angle_deg)
    c, s = np.cos(a), np.sin(a)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return world @ R.T


def _mirror(world: np.ndarray) -> np.ndarray:
    m = world.copy()
    for a, b in MIRROR_PAIRS:
        m[:, int(a)], m[:, int(b)] = world[:, int(b)].copy(), world[:, int(a)].copy()
    m[:, :, 0] *= -1.0
    return m


def _time_warp(world: np.ndarray, max_warp: float = 0.15) -> np.ndarray:
    T = world.shape[0]
    base = np.linspace(0.0, T - 1.0, T)
    rng = np.random.default_rng()
    perturbed = base.copy()
    perturbed[1:-1] += rng.normal(0.0, max_warp, size=T - 2)
    perturbed = np.sort(np.clip(perturbed, 0.0, T - 1.0))
    left = np.floor(perturbed).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (perturbed - left).reshape(-1, 1, 1)
    return (1 - alpha) * world[left] + alpha * world[right]


def _speed(world: np.ndarray, factor: float) -> np.ndarray:
    T = world.shape[0]
    new_T = max(8, int(round(T * factor)))
    idx = np.linspace(0, T - 1, new_T)
    left = np.floor(idx).astype(int)
    right = np.clip(left + 1, 0, T - 1)
    alpha = (idx - left).reshape(-1, 1, 1)
    sped = (1 - alpha) * world[left] + alpha * world[right]
    return _resample(sped, T)


_AUG_PER_CLASS = {0: 7, 1: 13}


def prepare(
    raw_dir: str,
    out_dir: str,
    n_frames: int = FRAMES_DEFAULT,
    seed: int = 42,
    test_size: float = 0.2,
    synth_jump: int = 0,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    cache_dir = os.path.join(out_dir, "cached")
    rng = np.random.default_rng(seed)

    all_paths: List[Tuple[str, int]] = []
    for name, label in LABEL_MAP.items():
        sub = os.path.join(raw_dir, name)
        if not os.path.isdir(sub):
            continue
        bvhs = list_bvh_files(sub)
        npys = sorted(
            os.path.join(sub, f) for f in os.listdir(sub) if f.endswith(".npy")
        )
        for p in bvhs + npys:
            all_paths.append((p, label))

    if not all_paths:
        raise FileNotFoundError(
            f"No .bvh/.npy files found under {raw_dir}/walk or {raw_dir}/jump."
        )

    print(
        f"Found {len(all_paths)} source sequences "
        f"({sum(1 for _, l in all_paths if l == 0)} walk, "
        f"{sum(1 for _, l in all_paths if l == 1)} jump)"
    )

    labels_all = np.array([l for _, l in all_paths])
    train_idx, test_idx = train_test_split(
        np.arange(len(all_paths)),
        test_size=test_size,
        random_state=seed,
        stratify=labels_all,
    )

    for split_name, idx_arr in [("train", train_idx), ("test", test_idx)]:
        seqs, labs, rests = [], [], []
        jump_base_worlds = []
        for i in idx_arr:
            path, label = all_paths[i]
            try:
                if path.endswith(".npy"):
                    world = np.load(path)
                else:
                    cached = os.path.join(cache_dir, os.path.basename(path) + ".npy")
                    world = load_cached(cached)
                    if world is None:
                        world = load_bvh_as_tensor(path)
                        save_cached(cached, world)
            except Exception as e:
                print(f"  skip {path}: {e}")
                continue
            if world.shape[0] < 4:
                continue
            world = _resample(world, n_frames).astype(np.float32)
            world = _center(world)
            if split_name == "train" and label == LABEL_MAP["jump"]:
                jump_base_worlds.append(world)

            variants = [world]
            if split_name == "train":
                for _ in range(_AUG_PER_CLASS[label]):
                    aug = world.copy()
                    if rng.random() < 0.5:
                        aug = _mirror(aug)
                    if rng.random() < 0.7:
                        aug = _time_warp(aug, 0.2)
                    if rng.random() < 0.7:
                        aug = _speed(aug, float(rng.uniform(0.85, 1.15)))
                    aug = _center(aug)
                    variants.append(aug)

            for w in variants:
                seqs.append(world_to_canonical(w).reshape(n_frames, FEATURE_DIM))
                rests.append(rest_bone_lengths(w))
                labs.append(label)

        # Synthetic jumps via motion interpolation
        if split_name == "train" and synth_jump > 0 and len(jump_base_worlds) >= 2:
            jump_rest = np.stack([rest_bone_lengths(w) for w in jump_base_worlds]).mean(
                axis=0
            )
            synth = synthesize_jumps(
                jump_base_worlds, synth_jump, jump_rest, seed=seed, n_frames=n_frames
            )
            for w in synth:
                seqs.append(world_to_canonical(w).reshape(n_frames, FEATURE_DIM))
                rests.append(rest_bone_lengths(w))
                labs.append(LABEL_MAP["jump"])
            print(f"  + {len(synth)} synthetic jump clips")

        seqs_np = np.stack(seqs).astype(np.float32)
        labs_np = np.array(labs, dtype=np.int64)
        rest_np = np.stack(rests).mean(axis=0).astype(np.float32)

        if split_name == "train":
            mean = seqs_np.mean(axis=(0, 1))
            std = np.maximum(seqs_np.std(axis=(0, 1)), 1e-2)
            np.savez(
                os.path.join(out_dir, "norm_stats.npz"),
                mean=mean,
                std=std,
                rest=rest_np,
            )
        else:
            stats = np.load(os.path.join(out_dir, "norm_stats.npz"))
            mean, std = stats["mean"], stats["std"]

        seqs_np = (seqs_np - mean) / std
        out_path = os.path.join(out_dir, f"{split_name}.npz")
        np.savez(out_path, sequences=seqs_np, labels=labs_np)
        print(
            f"  {split_name}: {seqs_np.shape}  labels={np.bincount(labs_np)}  "
            f"rest_meanlen={rest_np.mean():.3f}"
        )
        print(f"  → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=FRAMES_DEFAULT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument(
        "--synth-jump",
        type=int,
        default=0,
        help="number of synthetic jump clips to add (motion interpolation)",
    )
    args = ap.parse_args()
    prepare(
        args.raw_dir,
        args.out_dir,
        args.n_frames,
        args.seed,
        args.test_size,
        synth_jump=args.synth_jump,
    )


if __name__ == "__main__":
    main()
