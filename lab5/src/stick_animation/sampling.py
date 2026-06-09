from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch

from .diffusion import Diffusion
from .models.spatiotemporal_dit import FEATURE_DIM, SpatioTemporalDiT
from .representation import canonical_to_world


@torch.no_grad()
def sample(
    model: SpatioTemporalDiT,
    diffusion: Diffusion,
    class_label: int,
    n_samples: int,
    n_frames: int = 48,
    n_steps: int = 50,
    guidance_scale: float = 3.0,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    device = device or next(model.parameters()).device
    labels = torch.full((n_samples,), class_label, dtype=torch.long, device=device)
    shape = (n_samples, n_frames, FEATURE_DIM)
    return diffusion.ddim_sample(
        model,
        shape,
        labels,
        n_steps=n_steps,
        guidance_scale=guidance_scale,
        device=device,
    )


def feat_to_world(norm_feat: np.ndarray, norm: Dict[str, np.ndarray]) -> np.ndarray:
    """[N, T, 45] normalised canonical feature → [N, T, 15, 3] world positions.

    Per-channel denormalise → canonical_to_world (integrates root, faces +X) →
    bone-length snap to the stored rest skeleton (rigid, human proportions).
    Also accepts the legacy flat-world scalar stats (rest is None).
    """
    arr = np.asarray(norm_feat, dtype=np.float32)
    arr = arr * norm["std"] + norm["mean"]
    N, T, _ = arr.shape
    feat = arr.reshape(N, T, 15, 3)
    rest = norm.get("rest")
    if rest is None:
        return feat
    return np.stack([canonical_to_world(feat[i], rest_bones=rest) for i in range(N)])


def reconstruct(norm_samples: torch.Tensor, norm: Dict[str, np.ndarray]) -> np.ndarray:
    """DDIM output tensor → [N, T, 15, 3] world positions."""
    return feat_to_world(norm_samples.detach().cpu().numpy(), norm)
