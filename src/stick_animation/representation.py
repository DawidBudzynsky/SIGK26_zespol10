"""Motion representation conversions.

Key idea (lecture W12 / "Animacja postaci"):
the skeleton is a hierarchical tree rooted at the pelvis. Modelling joint
positions in world space — as the reference repo does — leaks two unrelated
signals into the same tensor:

    1. global translation + facing direction of the body
    2. local pose (how each joint sits w.r.t. its parent)

A diffusion model that has to learn both simultaneously, with no skeletal
prior, tends to produce poses with unstable bone lengths (the body
"breathes" between frames) and poor walk dynamics. We therefore decompose
the motion into:

    - root_xy_vel : per-frame planar velocity of the pelvis (2D)
    - root_z      : per-frame pelvis height (1D)
    - root_yaw    : per-frame facing direction (sin, cos) (2D)
    - local       : 14 non-root joints in pelvis-local frame (after yaw
                    cancellation), normalised by the average bone scale.

Forward kinematics reassembles a world-space [T, 15, 3] sequence from this
representation. Bone lengths are preserved by construction during sampling
because we additionally project sampled local joints onto their parent at
the average rest bone length (see ``snap_bone_lengths``).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import torch

from .skeleton import (
    BONES,
    JOINT_PARENTS,
    Joint,
    N_JOINTS,
)


# ----------------------------------------------------------------------------
# Pose decomposition / recomposition
# ----------------------------------------------------------------------------


def _yaw_from_shoulders(pose: np.ndarray) -> float:
    """Yaw (rad) that points the average shoulder line along +X.

    pose: [15, 3] in world space.
    """
    right = pose[int(Joint.RIGHT_SHOULDER)]
    left = pose[int(Joint.LEFT_SHOULDER)]
    side = right - left  # vector from left to right shoulder in XY
    # Body is facing perpendicular to the shoulder line, towards +X by convention.
    yaw = np.arctan2(side[1], side[0]) - np.pi / 2
    return float(yaw)


def _rotz(yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


@dataclass
class MotionRep:
    """Canonical motion representation.

    Shapes (T = number of frames):
        root_xy_vel : [T, 2]
        root_z      : [T, 1]
        root_yaw    : [T, 2]   (sin, cos)
        local       : [T, 14, 3]
        bone_scale  : float    (overall rest bone-length used for normalising)
    """

    root_xy_vel: np.ndarray
    root_z: np.ndarray
    root_yaw: np.ndarray
    local: np.ndarray
    bone_scale: float

    NON_ROOT = tuple(j for j in range(N_JOINTS) if j != int(Joint.PELVIS))

    @property
    def feature_dim(self) -> int:
        return 2 + 1 + 2 + 14 * 3  # 47

    def to_tensor(self) -> np.ndarray:
        """Concatenate into a single [T, F] tensor for the network."""
        T = self.root_xy_vel.shape[0]
        flat_local = self.local.reshape(T, -1)
        return np.concatenate(
            [self.root_xy_vel, self.root_z, self.root_yaw, flat_local], axis=-1
        ).astype(np.float32)

    @classmethod
    def from_tensor(cls, t: np.ndarray, bone_scale: float) -> "MotionRep":
        T = t.shape[0]
        root_xy_vel = t[:, 0:2]
        root_z = t[:, 2:3]
        root_yaw = t[:, 3:5]
        local = t[:, 5:].reshape(T, 14, 3)
        return cls(root_xy_vel, root_z, root_yaw, local, bone_scale)


def world_to_rep(world: np.ndarray) -> MotionRep:
    """[T, 15, 3] world-space positions → canonical MotionRep."""
    world = np.asarray(world, dtype=np.float32)
    T = world.shape[0]

    pelvis = world[:, int(Joint.PELVIS)]  # [T, 3]
    root_xy = pelvis[:, :2]
    root_z = pelvis[:, 2:3]

    # Frame-wise yaw aligns the body to face +X.
    yaws = np.array([_yaw_from_shoulders(world[t]) for t in range(T)], dtype=np.float32)
    yaws = np.unwrap(yaws)  # smooth across frames
    root_yaw = np.stack([np.sin(yaws), np.cos(yaws)], axis=-1)

    # Local positions: subtract pelvis, then rotate by -yaw around Z.
    local = np.zeros((T, 14, 3), dtype=np.float32)
    non_root = [j for j in range(N_JOINTS) if j != int(Joint.PELVIS)]
    for t in range(T):
        R_inv = _rotz(-yaws[t])
        centered = world[t, non_root] - pelvis[t]
        local[t] = centered @ R_inv.T

    # Bone scale: mean bone length in the rest (mean) pose, used to normalise
    # joint magnitudes so the network sees ~unit-scale features.
    mean_pose = world.mean(axis=0)
    bones = np.linalg.norm(
        np.stack([mean_pose[c] - mean_pose[p] for c, p in BONES], axis=0),
        axis=-1,
    )
    bone_scale = float(bones.mean()) if bones.mean() > 1e-6 else 1.0
    local /= bone_scale

    # Velocity of root xy (last frame replicates penultimate to keep length T).
    root_xy_vel = np.zeros_like(root_xy)
    root_xy_vel[1:] = root_xy[1:] - root_xy[:-1]
    root_xy_vel[0] = root_xy_vel[1]
    root_xy_vel /= bone_scale

    root_z = root_z / bone_scale

    return MotionRep(root_xy_vel, root_z, root_yaw, local, bone_scale)


def rep_to_world(rep: MotionRep, start_xy: Tuple[float, float] = (0.0, 0.0)) -> np.ndarray:
    """MotionRep → [T, 15, 3] world-space positions."""
    T = rep.root_xy_vel.shape[0]
    s = rep.bone_scale

    xy = np.zeros((T, 2), dtype=np.float32)
    xy[0] = np.array(start_xy, dtype=np.float32)
    for t in range(1, T):
        xy[t] = xy[t - 1] + rep.root_xy_vel[t] * s
    z = (rep.root_z * s).reshape(T, 1)
    pelvis = np.concatenate([xy, z], axis=-1)  # [T, 3]

    sin_y, cos_y = rep.root_yaw[:, 0], rep.root_yaw[:, 1]
    # Renormalise (sin, cos) — the model output may drift.
    norm = np.sqrt(sin_y ** 2 + cos_y ** 2) + 1e-6
    sin_y /= norm
    cos_y /= norm
    yaws = np.arctan2(sin_y, cos_y)

    non_root = [j for j in range(N_JOINTS) if j != int(Joint.PELVIS)]
    world = np.zeros((T, N_JOINTS, 3), dtype=np.float32)
    for t in range(T):
        R = _rotz(yaws[t])
        local_world = rep.local[t] * s @ R.T + pelvis[t]
        world[t, int(Joint.PELVIS)] = pelvis[t]
        for k, j in enumerate(non_root):
            world[t, j] = local_world[k]
    return world


# ----------------------------------------------------------------------------
# Bone-length preservation (used on generated samples).
# ----------------------------------------------------------------------------


def rest_bone_lengths(motion: np.ndarray) -> np.ndarray:
    """Per-bone rest length in the *mean* pose. motion: [T, 15, 3]."""
    mean_pose = motion.mean(axis=0)
    return np.array(
        [float(np.linalg.norm(mean_pose[c] - mean_pose[p])) for c, p in BONES],
        dtype=np.float32,
    )


def snap_bone_lengths(world: np.ndarray, rest: np.ndarray) -> np.ndarray:
    """Project joints so every bone has its rest length while keeping direction.

    Traverses the kinematic tree from PELVIS down so children inherit
    corrected parent positions.

    world: [T, 15, 3], rest: [n_bones].
    """
    T = world.shape[0]
    out = world.copy().astype(np.float32)
    # Iterate BFS from root.
    bone_index = {c: i for i, (c, _) in enumerate(BONES)}
    visited = {int(Joint.PELVIS)}
    queue = [int(Joint.PELVIS)]
    while queue:
        parent = queue.pop(0)
        for c, p in BONES:
            if p == parent and c not in visited:
                dir_ = out[:, c] - out[:, p]
                length = np.linalg.norm(dir_, axis=-1, keepdims=True) + 1e-8
                dir_ = dir_ / length
                out[:, c] = out[:, p] + dir_ * rest[bone_index[c]]
                visited.add(c)
                queue.append(c)
    return out


# ----------------------------------------------------------------------------
# DCT helpers — low-frequency motion features (PoseFormerV2-inspired).
# ----------------------------------------------------------------------------


def dct_matrix(T: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of size [T, T]."""
    n = np.arange(T)[None, :]
    k = np.arange(T)[:, None]
    M = np.cos(np.pi * (2 * n + 1) * k / (2 * T))
    M = M * np.sqrt(2.0 / T)
    M[0] *= 1.0 / np.sqrt(2.0)
    return M.astype(np.float32)


def torch_dct(x: "torch.Tensor") -> "torch.Tensor":
    """DCT-II along the time axis. x: [B, T, ...]."""
    import torch
    T = x.shape[1]
    M = torch.tensor(dct_matrix(T), device=x.device, dtype=x.dtype)
    flat = x.reshape(x.shape[0], T, -1)
    out = torch.einsum("kt,btd->bkd", M, flat)
    return out.reshape(x.shape)


def torch_idct(x: "torch.Tensor") -> "torch.Tensor":
    """Inverse DCT-II along the time axis."""
    import torch
    T = x.shape[1]
    M = torch.tensor(dct_matrix(T), device=x.device, dtype=x.dtype)
    flat = x.reshape(x.shape[0], T, -1)
    out = torch.einsum("kt,bkd->btd", M, flat)
    return out.reshape(x.shape)
