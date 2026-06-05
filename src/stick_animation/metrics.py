"""Quantitative metrics required by the project brief: FMD, MPJPE and a
diversity (Variance) score, plus a few extra diagnostic numbers."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import linalg

from .skeleton import Joint, N_JOINTS


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------


def _root_relative(motion: np.ndarray) -> np.ndarray:
    """Subtract pelvis (root) trajectory at every frame so the metric is
    independent of where the body happens to be in the world. motion: [N, T, 15, 3]."""
    pelvis = motion[:, :, int(Joint.PELVIS) : int(Joint.PELVIS) + 1]
    return motion - pelvis


def motion_features(motion: np.ndarray, root_relative: bool = True) -> np.ndarray:
    """Per-sample feature vector used by FMD.

    [mean_pose | std_pose | mean_vel | std_vel] flattened to 1D.
    """
    if root_relative:
        motion = _root_relative(motion)
    N = motion.shape[0]
    vel = np.diff(motion, axis=1)
    feats = [
        motion.mean(axis=1).reshape(N, -1),
        motion.std(axis=1).reshape(N, -1),
        vel.mean(axis=1).reshape(N, -1),
        vel.std(axis=1).reshape(N, -1),
    ]
    return np.concatenate(feats, axis=1).astype(np.float64)


# ----------------------------------------------------------------------------
# Frechet Motion Distance
# ----------------------------------------------------------------------------


def _frechet(mu1, sig1, mu2, sig2, eps: float = 1e-6) -> float:
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sig1 @ sig2, disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sig1.shape[0]) * eps
        covmean = linalg.sqrtm((sig1 + offset) @ (sig2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    return float(diff @ diff + np.trace(sig1 + sig2 - 2 * covmean))


def compute_fmd(real: np.ndarray, generated: np.ndarray, root_relative: bool = True) -> float:
    fr = motion_features(real, root_relative)
    fg = motion_features(generated, root_relative)
    mu_r, sig_r = fr.mean(0), np.cov(fr, rowvar=False)
    mu_g, sig_g = fg.mean(0), np.cov(fg, rowvar=False)
    return _frechet(mu_r, sig_r, mu_g, sig_g)


# ----------------------------------------------------------------------------
# MPJPE
# ----------------------------------------------------------------------------


def compute_mpjpe(real: np.ndarray, generated: np.ndarray,
                  root_relative: bool = True) -> float:
    """Mean per-joint position error against the nearest-neighbour real motion.

    For every generated sample we find its nearest real sample in feature
    space, then compute the average Euclidean per-joint per-frame distance.
    Root-relative by default so the score reflects pose quality, not global
    drift.
    """
    if root_relative:
        real = _root_relative(real)
        generated = _root_relative(generated)
    fr = motion_features(real, root_relative=False)
    fg = motion_features(generated, root_relative=False)

    # Match generated → real by L2 in feature space.
    fr_norm = (fr ** 2).sum(1)
    fg_norm = (fg ** 2).sum(1)
    dists = fr_norm[None, :] + fg_norm[:, None] - 2 * fg @ fr.T
    nn = dists.argmin(axis=1)
    errors = []
    for i, j in enumerate(nn):
        diff = generated[i] - real[j]                    # [T, 15, 3]
        per_joint = np.linalg.norm(diff, axis=-1).mean()
        errors.append(per_joint)
    return float(np.mean(errors))


# ----------------------------------------------------------------------------
# Variance / diversity
# ----------------------------------------------------------------------------


def compute_variance(generated: np.ndarray) -> Dict[str, float]:
    """Pairwise feature distances + joint position std + velocity std."""
    feats = motion_features(generated)
    N = feats.shape[0]
    if N < 2:
        return {"pairwise_dist": 0.0, "joint_std": 0.0, "vel_std": 0.0}
    pairs = []
    for i in range(N):
        for j in range(i + 1, N):
            pairs.append(np.linalg.norm(feats[i] - feats[j]))
    vel = np.diff(generated, axis=1)
    return {
        "pairwise_dist": float(np.mean(pairs)),
        "joint_std": float(generated.std(axis=0).mean()),
        "vel_std": float(vel.std(axis=0).mean()),
    }


# ----------------------------------------------------------------------------
# Aggregate report
# ----------------------------------------------------------------------------


def summarise(real: np.ndarray, generated: np.ndarray) -> Dict[str, float]:
    fmd = compute_fmd(real, generated)
    mpjpe = compute_mpjpe(real, generated)
    var = compute_variance(generated)
    return {"FMD": fmd, "MPJPE": mpjpe, **{f"Var_{k}": v for k, v in var.items()}}
