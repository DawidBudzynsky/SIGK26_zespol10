import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResidualBlock(nn.Module):

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return out


class UpsampleBlock(nn.Module):

    def __init__(self, channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(
            channels, channels * (scale_factor**2), kernel_size=3, padding=1
        )
        self.shuffle = nn.PixelShuffle(scale_factor)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.shuffle(out)
        return self.relu(out)


class UpscaleNet(nn.Module):
    """Super-resolution network.

    Architecture:
    - Entry conv
    - Residual blocks
    - Upsample blocks (using pixel shuffle)
    - Exit conv
    - Residual connection with bicubic upsampled input
    """

    def __init__(self, num_residual_blocks=8, channels=64, upscale_factor=4):
        super(UpscaleNet, self).__init__()
        self.upscale_factor = upscale_factor
        self.entry_conv = nn.Conv2d(3, channels, kernel_size=9, padding=4)

        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        num_upsample = int(math.log2(upscale_factor))
        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(channels, scale_factor=2) for _ in range(num_upsample)]
        )

        self.exit_conv = nn.Conv2d(channels, 3, kernel_size=9, padding=4)

        self.activation = nn.Tanh()

    def forward(self, x):
        out = F.relu(self.entry_conv(x))
        out = self.residual_blocks(out)
        out = self.upsample_blocks(out)
        out = self.exit_conv(out)

        residual = F.interpolate(
            x, scale_factor=self.upscale_factor, mode="bicubic", align_corners=False
        )
        out = out + residual

        out = (out + 1) / 2
        out = torch.clamp(out, 0, 1)

        return out
