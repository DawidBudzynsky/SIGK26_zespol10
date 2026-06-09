from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def cosine_alphas_cumprod(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    t = torch.linspace(0, 1, steps)
    a = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
    a = a / a[0]
    betas = 1 - a[1:] / a[:-1]
    betas = betas.clamp(0.0001, 0.9999)
    alphas = 1.0 - betas
    return torch.cumprod(alphas, dim=0)


def linear_alphas_cumprod(
    timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02
) -> torch.Tensor:
    betas = torch.linspace(beta_start, beta_end, timesteps)
    return torch.cumprod(1.0 - betas, dim=0)


class Diffusion:
    def __init__(
        self,
        timesteps: int = 1000,
        schedule: str = "cosine",
        parametrization: str = "v",
    ):
        self.T = timesteps
        self.parametrization = parametrization
        if schedule == "cosine":
            alphas_cumprod = cosine_alphas_cumprod(timesteps)
        elif schedule == "linear":
            alphas_cumprod = linear_alphas_cumprod(timesteps)
        else:
            raise ValueError(f"Unknown schedule {schedule}")
        self.alphas_cumprod = alphas_cumprod
        self.sqrt_alpha = alphas_cumprod.sqrt()
        self.sqrt_one_minus = (1.0 - alphas_cumprod).sqrt()

    def to(self, device) -> "Diffusion":
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        self.sqrt_alpha = self.sqrt_alpha.to(device)
        self.sqrt_one_minus = self.sqrt_one_minus.to(device)
        return self

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor):
        sqrt_a = self.sqrt_alpha[t].reshape(-1, 1, 1)
        sqrt_1a = self.sqrt_one_minus[t].reshape(-1, 1, 1)
        return sqrt_a * x0 + sqrt_1a * noise

    def v_target(self, x0: torch.Tensor, noise: torch.Tensor, t: torch.Tensor):
        sqrt_a = self.sqrt_alpha[t].reshape(-1, 1, 1)
        sqrt_1a = self.sqrt_one_minus[t].reshape(-1, 1, 1)
        return sqrt_a * noise - sqrt_1a * x0

    def x0_from_v(self, v_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        sqrt_a = self.sqrt_alpha[t].reshape(-1, 1, 1)
        sqrt_1a = self.sqrt_one_minus[t].reshape(-1, 1, 1)
        return sqrt_a * x_t - sqrt_1a * v_pred

    def eps_from_v(self, v_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        sqrt_a = self.sqrt_alpha[t].reshape(-1, 1, 1)
        sqrt_1a = self.sqrt_one_minus[t].reshape(-1, 1, 1)
        return sqrt_a * v_pred + sqrt_1a * x_t

    def x0_from_eps(self, eps_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        sqrt_a = self.sqrt_alpha[t].reshape(-1, 1, 1)
        sqrt_1a = self.sqrt_one_minus[t].reshape(-1, 1, 1)
        return (x_t - sqrt_1a * eps_pred) / sqrt_a.clamp(min=1e-8)

    def training_pred(
        self,
        model: nn.Module,
        x0: torch.Tensor,
        t: torch.Tensor,
        class_label: torch.Tensor,
        cfg_drop_prob: float = 0.1,
    ):
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)

        mask = (
            torch.rand(class_label.shape[0], device=class_label.device) < cfg_drop_prob
        )
        label_input = class_label.clone()
        label_input[mask] = model.null_class_idx

        pred = model(x_t, t, label_input)

        if self.parametrization == "v":
            target = self.v_target(x0, noise, t)
        elif self.parametrization == "eps":
            target = noise
        else:
            raise ValueError(self.parametrization)
        return target, pred, x_t, noise, label_input

    def x0_from_pred(self, pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor):
        if self.parametrization == "v":
            return self.x0_from_v(pred, x_t, t)
        return self.x0_from_eps(pred, x_t, t)

    @torch.no_grad()
    def ddim_sample(
        self,
        model: nn.Module,
        shape: tuple,
        class_label: torch.Tensor,
        n_steps: int = 50,
        guidance_scale: float = 3.0,
        eta: float = 0.0,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        device = device or next(model.parameters()).device
        x = torch.randn(shape, device=device)

        step_idx = torch.linspace(0, self.T - 1, n_steps).long().to(device)
        step_idx = torch.flip(step_idx, dims=[0])

        null_label = torch.full_like(class_label, model.null_class_idx)
        for i, t in enumerate(step_idx):
            t_b = torch.full((shape[0],), int(t), device=device, dtype=torch.long)

            pred_cond = model(x, t_b, class_label)
            pred_uncond = model(x, t_b, null_label)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            x0_pred = self.x0_from_pred(pred, x, t_b)
            if self.parametrization == "v":
                eps_pred = self.eps_from_v(pred, x, t_b)
            else:
                eps_pred = pred

            if i < len(step_idx) - 1:
                t_prev = step_idx[i + 1]
                a_t = self.alphas_cumprod[t]
                a_prev = self.alphas_cumprod[t_prev]
                sigma = (
                    eta * ((1 - a_prev) / (1 - a_t)).sqrt() * (1 - a_t / a_prev).sqrt()
                )
                dir_xt = (1 - a_prev - sigma**2).clamp(min=0).sqrt() * eps_pred
                x = a_prev.sqrt() * x0_pred + dir_xt
                if eta > 0:
                    x = x + sigma * torch.randn_like(x)
            else:
                x = x0_pred
        return x


def primary_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    """MSE as main diffusion loss"""
    return F.mse_loss(pred, target)
