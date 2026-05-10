from __future__ import annotations
import torch
import torch.nn as nn

PARAM_DIM = 10
IMG_SIZE = 128


def _init(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.normal_(m.weight, 0.0, 0.02)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


def _coord_grid(size: int, device: torch.device) -> torch.Tensor:
    y = (
        torch.linspace(-1, 1, size, device=device)
        .view(1, 1, size, 1)
        .expand(1, 1, size, size)
    )
    x = (
        torch.linspace(-1, 1, size, device=device)
        .view(1, 1, 1, size)
        .expand(1, 1, size, size)
    )
    return torch.cat([x, y], dim=1)


class Generator(nn.Module):
    def __init__(self, param_dim: int = PARAM_DIM, base: int = 96, n_refine: int = 3):
        super().__init__()
        self.param_dim = param_dim

        # per-pixel mlp -> realizowany 1x1 konwolucjami; każdy piksel widzi tylko siebie (jego (x, y) + parametry sceny) i decyduje o kolorze.
        in_c = 2 + param_dim
        self.pixel_mlp = nn.Sequential(
            nn.Conv2d(in_c, base * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base * 2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base * 2, base, 1),
            nn.ReLU(inplace=True),
        )

        # spatial refinement, 3x3 conv + InstanceNorm + ReLU * n_refine, dodaje przestrzenną spójność (krawędzie kuli, gradient cieniowania).
        refine = []
        for _ in range(n_refine):
            refine += [
                nn.Conv2d(base, base, 3, padding=1),
                nn.InstanceNorm2d(base, affine=True),
                nn.ReLU(inplace=True),
            ]
        self.refine = nn.Sequential(*refine)

        self.to_rgb = nn.Sequential(
            nn.Conv2d(base, base // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base // 2, 3, 1),
            nn.Sigmoid(),
        )

        self.apply(_init)

    def forward(self, params: torch.Tensor) -> torch.Tensor:
        b = params.size(0)
        coords = _coord_grid(IMG_SIZE, params.device).expand(b, -1, -1, -1)
        p_map = params.view(b, -1, 1, 1).expand(b, -1, IMG_SIZE, IMG_SIZE)
        x = torch.cat([coords, p_map], dim=1)  # (B, 12, 128, 128)
        x = self.pixel_mlp(x)
        x = self.refine(x)
        return self.to_rgb(x)


class Discriminator(nn.Module):
    def __init__(self, param_dim: int = PARAM_DIM, base: int = 64):
        super().__init__()
        in_c = 3 + param_dim

        def block(c_in, c_out, stride=2, norm=True):
            layers = [
                nn.Conv2d(c_in, c_out, 4, stride=stride, padding=1, bias=not norm)
            ]
            if norm:
                layers.append(nn.InstanceNorm2d(c_out, affine=True))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        self.body = nn.Sequential(
            block(in_c, base, norm=False),
            block(base, base * 2),
            block(base * 2, base * 4),
            block(base * 4, base * 8, stride=1),
            nn.Conv2d(base * 8, 1, 4, stride=1, padding=1),
        )
        self.apply(_init)

    def forward(self, img, params):
        b, _, h, w = img.shape
        p_map = params.view(b, -1, 1, 1).expand(b, -1, h, w)
        return self.body(torch.cat([img, p_map], dim=1))
