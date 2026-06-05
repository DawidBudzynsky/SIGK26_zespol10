"""Multi-objective loss following the energy-minimization view from lecture W12.

The training signal mixes:

    *  ``L_v``       — primary diffusion loss on the chosen parametrization
                       (v- or ε-prediction).
    *  ``L_bone``    — MSE between predicted bone lengths and the rest bone
                       lengths. Directly counteracts the most visible
                       artefact in the reference walk samples (bones
                       stretching/contracting between frames).
    *  ``L_smooth``  — acceleration penalty (E_smooth in VNect). Penalises
                       jitter and the high-frequency noise that DDPM
                       produces near the end of the reverse process.
    *  ``L_foot``    — foot-skating loss. The contact-gated horizontal
                       velocity of the ankle, computed in world space after
                       FK reconstruction.
    *  ``L_vel``     — velocity match between predicted and target on the
                       diffusion parametrization (cheap regulariser already
                       used by the reference repo).

All extra terms operate on the predicted x0 (decoded from the parametrized
prediction) so they tie back to the geometry the user actually cares about.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from .models.spatiotemporal_dit import LOCAL_PER_JOINT, N_LOCAL_JOINTS, ROOT_FEATS
from .skeleton import BONES, JOINT_PARENTS, Joint, N_JOINTS

# Bone endpoints expressed in the "local" / non-root index space.
# A local index i ∈ [0, 13] corresponds to global joint i+1 if i+1 != PELVIS,
# otherwise i (which never happens because PELVIS=2 is root by construction).
# We just build a lookup once.
_LOCAL_OF = [None] * N_JOINTS
_li = 0
for _j in range(N_JOINTS):
    if _j != int(Joint.PELVIS):
        _LOCAL_OF[_j] = _li
        _li += 1


@dataclass
class LossWeights:
    bone: float = 0.5
    smooth: float = 0.05
    foot: float = 0.2
    vel: float = 0.1


# ----------------------------------------------------------------------------
# Differentiable representation helpers
# ----------------------------------------------------------------------------


def _denormalise(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    """x: [..., F]. mean, std: [F]."""
    return x * std + mean


def _split_rep(x: torch.Tensor):
    """Split a [B, T, F] tensor into root and local sub-tensors."""
    root = x[..., :ROOT_FEATS]
    local = x[..., ROOT_FEATS:].reshape(*x.shape[:-1], N_LOCAL_JOINTS, LOCAL_PER_JOINT)
    return root, local


def world_from_rep(
    x_norm: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    bone_scale: torch.Tensor,
) -> torch.Tensor:
    """Differentiable inverse of ``world_to_rep`` for a batch.

    x_norm:    [B, T, F] normalised features
    mean/std:  [F]
    bone_scale:[B]
    returns world: [B, T, 15, 3]
    """
    x = _denormalise(x_norm, mean, std)
    root, local = _split_rep(x)
    B, T, _ = root.shape
    s = bone_scale.reshape(B, 1, 1)

    # Pelvis position by cumulative sum of velocity.
    xy_vel = root[..., :2] * s
    z = root[..., 2:3] * s
    xy = torch.cumsum(xy_vel, dim=1)  # [B, T, 2] (starts at 0)
    pelvis = torch.cat([xy, z], dim=-1)  # [B, T, 3]

    # Yaw rotation matrix.
    sin_y, cos_y = root[..., 3], root[..., 4]
    norm = torch.sqrt(sin_y ** 2 + cos_y ** 2).clamp(min=1e-6)
    sin_y = sin_y / norm
    cos_y = cos_y / norm
    # R rotates local→world: [[c, -s, 0], [s, c, 0], [0, 0, 1]]

    local_scaled = local * s.unsqueeze(-1)  # [B, T, 14, 3]
    lx = local_scaled[..., 0]
    ly = local_scaled[..., 1]
    lz = local_scaled[..., 2]
    wx = cos_y.unsqueeze(-1) * lx - sin_y.unsqueeze(-1) * ly
    wy = sin_y.unsqueeze(-1) * lx + cos_y.unsqueeze(-1) * ly
    wz = lz
    local_world = torch.stack([wx, wy, wz], dim=-1) + pelvis.unsqueeze(2)

    world = torch.empty(B, T, N_JOINTS, 3, device=x.device, dtype=x.dtype)
    world[:, :, int(Joint.PELVIS)] = pelvis
    non_root = [j for j in range(N_JOINTS) if j != int(Joint.PELVIS)]
    for li, j in enumerate(non_root):
        world[:, :, j] = local_world[:, :, li]
    return world


# ----------------------------------------------------------------------------
# Loss terms
# ----------------------------------------------------------------------------


def bone_length_loss(world: torch.Tensor, rest_bones: torch.Tensor) -> torch.Tensor:
    """world: [B, T, 15, 3]. rest_bones: [B, n_bones]."""
    # Compute per-bone length per frame.
    parts = []
    for c, p in BONES:
        d = world[:, :, c] - world[:, :, p]
        parts.append(torch.linalg.norm(d, dim=-1))  # [B, T]
    lengths = torch.stack(parts, dim=-1)  # [B, T, n_bones]
    target = rest_bones.unsqueeze(1)      # [B, 1, n_bones]
    return F.mse_loss(lengths, target.expand_as(lengths))


def smoothness_loss(world: torch.Tensor) -> torch.Tensor:
    """Acceleration penalty over time. world: [B, T, 15, 3]."""
    acc = world[:, 2:] - 2 * world[:, 1:-1] + world[:, :-2]
    return (acc ** 2).mean()


def foot_skating_loss(
    world: torch.Tensor,
    contact_height: float = 0.05,
    soft_temp: float = 10.0,
) -> torch.Tensor:
    """Penalise horizontal ankle velocity when the ankle is close to floor.

    Floor estimated per-sequence as the min ankle z. Contact gate is a soft
    sigmoid so the loss is differentiable.
    """
    ankles = torch.stack(
        [world[:, :, int(Joint.RIGHT_ANKLE)], world[:, :, int(Joint.LEFT_ANKLE)]],
        dim=2,
    )  # [B, T, 2, 3]
    floor = ankles[..., 2].amin(dim=(1, 2), keepdim=True)  # [B, 1, 1]
    height_above = ankles[..., 2] - floor                  # [B, T, 2]
    contact = torch.sigmoid(soft_temp * (contact_height - height_above))

    vel = ankles[:, 1:, :, :2] - ankles[:, :-1, :, :2]     # [B, T-1, 2, 2]
    speed_sq = (vel ** 2).sum(dim=-1)                      # [B, T-1, 2]
    gate = 0.5 * (contact[:, 1:] + contact[:, :-1])
    return (gate * speed_sq).mean()


def velocity_match_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """MSE on temporal differences of target vs prediction."""
    tv = target[:, 1:] - target[:, :-1]
    pv = pred[:, 1:] - pred[:, :-1]
    return F.mse_loss(pv, tv)


def total_loss(
    target_param: torch.Tensor,
    pred_param: torch.Tensor,
    x0_pred_norm: torch.Tensor,
    rest_bones: torch.Tensor,
    bone_scale: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    weights: LossWeights = LossWeights(),
    use_world_losses: bool = True,
) -> Dict[str, torch.Tensor]:
    """Compose primary + geometric losses. Returns dict with per-term values.

    target_param, pred_param: shape [B, T, F] (v or eps prediction space).
    x0_pred_norm: shape [B, T, F] decoded back to x0 in normalised space.
    rest_bones, bone_scale: per-sample geometry.
    """
    out: Dict[str, torch.Tensor] = {}
    primary = F.mse_loss(pred_param, target_param)
    out["primary"] = primary
    out["velocity"] = velocity_match_loss(target_param, pred_param)
    total = primary + weights.vel * out["velocity"]

    if use_world_losses:
        world = world_from_rep(x0_pred_norm, mean, std, bone_scale)
        out["bone"] = bone_length_loss(world, rest_bones)
        out["smooth"] = smoothness_loss(world)
        out["foot"] = foot_skating_loss(world)
        total = (
            total
            + weights.bone * out["bone"]
            + weights.smooth * out["smooth"]
            + weights.foot * out["foot"]
        )
    out["total"] = total
    return out
