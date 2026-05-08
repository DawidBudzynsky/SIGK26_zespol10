"""Generator + Discriminator for the conditional Phong-renderer GAN.

Generator: takes a 15-dim scene parameter vector and produces a 128x128 RGB
image. Architecture is a small ConvTranspose decoder seeded by an MLP that
projects the scene parameters into a 4x4x256 spatial latent.

Discriminator: PatchGAN-style 70x70 patch discriminator that conditions on
the parameters (broadcast as extra channels) so it can tell real-from-fake
*for that scene* rather than just "is this any plausible Phong sphere".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


PARAM_DIM = 15


class Generator(nn.Module):
    def __init__(self, param_dim: int = PARAM_DIM, base: int = 64):
        super().__init__()
        self.param_dim = param_dim

        # Project params -> 4x4x(base*8) spatial latent
        self.fc = nn.Sequential(
            nn.Linear(param_dim, base * 8),
            nn.ReLU(inplace=True),
            nn.Linear(base * 8, base * 8 * 4 * 4),
            nn.ReLU(inplace=True),
        )
        self.base = base

        def up(in_c, out_c):
            return nn.Sequential(
                nn.ConvTranspose2d(in_c, out_c, 4, stride=2, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

        self.up1 = up(base * 8, base * 8)   # 4 -> 8
        self.up2 = up(base * 8, base * 4)   # 8 -> 16
        self.up3 = up(base * 4, base * 2)   # 16 -> 32
        self.up4 = up(base * 2, base)       # 32 -> 64
        self.up5 = up(base, base // 2)      # 64 -> 128

        self.out = nn.Sequential(
            nn.Conv2d(base // 2, 3, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        x = self.fc(params)
        x = x.view(-1, self.base * 8, 4, 4)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.up4(x)
        x = self.up5(x)
        return self.out(x)


class Discriminator(nn.Module):
    """Conditional PatchGAN discriminator.

    Concatenates the scene parameters as constant feature maps to the input
    image so the discriminator can score authenticity *given the parameters*.
    """
    def __init__(self, param_dim: int = PARAM_DIM, base: int = 64):
        super().__init__()
        in_c = 3 + param_dim

        def conv(in_c, out_c, stride=2, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, stride=stride, padding=1)]
            if norm:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.net = nn.Sequential(
            conv(in_c, base, norm=False),    # 128 -> 64
            conv(base, base * 2),            # 64 -> 32
            conv(base * 2, base * 4),        # 32 -> 16
            conv(base * 4, base * 8, stride=1),  # 16 -> 15
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),  # 15 -> 14
        )

    def forward(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        b, c, h, w = img.shape
        p = params.view(b, -1, 1, 1).expand(b, -1, h, w)
        x = torch.cat([img, p], dim=1)
        return self.net(x)
