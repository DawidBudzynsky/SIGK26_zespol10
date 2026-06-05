"""Sampling pipeline: DDIM → denormalise → reconstruct world positions → bone snap."""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch

from .diffusion import Diffusion
from .models.spatiotemporal_dit import SpatioTemporalDiT
from .representation import MotionRep, rep_to_world, snap_bone_lengths


@torch.no_grad()
def sample(
    model: SpatioTemporalDiT,
    diffusion: Diffusion,
    class_label: int,
    n_samples: int,
    feature_dim: int,
    n_frames: int = 48,
    n_steps: int = 50,
    guidance_scale: float = 3.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    labels = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    shape = (n_samples, n_frames, feature_dim)
    return diffusion.ddim_sample(model, shape, labels,
                                 n_steps=n_steps, guidance_scale=guidance_scale,
                                 device=device)


def reconstruct(
    norm_samples: torch.Tensor,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    rest_bones: np.ndarray,
    bone_scale: float = 1.0,
    apply_bone_snap: bool = True,
) -> np.ndarray:
    """Take model output (normalised canonical features) → world [N, T, 15, 3]."""
    arr = norm_samples.detach().cpu().numpy()
    arr = arr * norm_std + norm_mean  # de-normalise

    worlds = []
    for s in arr:
        rep = MotionRep.from_tensor(s, bone_scale=bone_scale)
        world = rep_to_world(rep)
        if apply_bone_snap:
            world = snap_bone_lengths(world, rest_bones * bone_scale)
        worlds.append(world)
    return np.stack(worlds, axis=0)
