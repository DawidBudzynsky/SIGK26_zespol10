import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class ExposureSynthesizer(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 6):
        """
        out_channels=6: channels 0:3 underexposed (-2.7 EV), channels 3:6 overexposed (+2.7 EV)
        """
        super().__init__()
        
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        
        self.up1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        
        self.up2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        
        self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        
        self.up4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        
        self.outc = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        return self._forward_impl(x1, x2, x3, x4, x5)

    def _forward_impl(self, x1: torch.Tensor, x2: torch.Tensor, 
                      x3: torch.Tensor, x4: torch.Tensor, x5: torch.Tensor) -> torch.Tensor:
        
        d1 = self.up1(x5)
        d1 = self._pad_and_cat(d1, x4)
        d1 = self.conv1(d1)
        
        d2 = self.up2(d1)
        d2 = self._pad_and_cat(d2, x3)
        d2 = self.conv2(d2)
        
        d3 = self.up3(d2)
        d3 = self._pad_and_cat(d3, x2)
        d3 = self.conv3(d3)
        
        d4 = self.up4(d3)
        d4 = self._pad_and_cat(d4, x1)
        d4 = self.conv4(d4)
        
        return torch.sigmoid(self.outc(d4))
    
    def _pad_and_cat(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        diffY = skip.size(2) - x.size(2)
        diffX = skip.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])
        return torch.cat([skip, x], dim=1)