from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights


class VGGPerceptualLoss(nn.Module):
    """L1 w przestrzeni cech VGG16 (ImageNet-pretrained), warstwy relu1_2, relu2_2, relu3_3.

    Karze za niedopasowanie *kształtu* i *lokalizacji* obiektu, nie tylko
    piksel-w-piksel — wybawia z lokalnego minimum 'rozmytej średniej', do
    którego L1/MSE zbiega pod niepewnością pozycji.
    """

    def __init__(self):
        super().__init__()
        feats = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features.eval()
        # punkty cięcia obejmujące relu1_2 (3), relu2_2 (8), relu3_3 (15)
        self.slice1 = nn.Sequential(*[feats[i] for i in range(4)])
        self.slice2 = nn.Sequential(*[feats[i] for i in range(4, 9)])
        self.slice3 = nn.Sequential(*[feats[i] for i in range(9, 16)])
        for p in self.parameters():
            p.requires_grad_(False)

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def _norm(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def forward(self, fake: torch.Tensor, real: torch.Tensor) -> torch.Tensor:
        x = self._norm(fake)
        y = self._norm(real)
        loss = torch.zeros((), device=fake.device)
        for s in (self.slice1, self.slice2, self.slice3):
            x = s(x)
            y = s(y)
            loss = loss + F.l1_loss(x, y)
        return loss
