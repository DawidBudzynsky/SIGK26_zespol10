"""Training losses for flat world-space representation [T, 45]."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def velocity_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    tv = target[:, 1:] - target[:, :-1]
    pv = pred[:, 1:] - pred[:, :-1]
    return F.mse_loss(pv, tv)


def total_loss(target: torch.Tensor, pred: torch.Tensor,
               vel_weight: float = 0.1) -> torch.Tensor:
    primary = F.mse_loss(pred, target)
    vel = velocity_loss(target, pred)
    return primary + vel_weight * vel, {"primary": primary, "velocity": vel}
