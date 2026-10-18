"""
ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation.

Reference:
    Paszke, A., Chaurasia, A., Kim, S., & Culurciello, E. (2016).
    ENet: A Deep Neural Network Architecture for Real-Time Semantic Segmentation.
    arXiv:1606.02147.
    Repository: https://github.com/davidtvs/PyTorch-ENet
    Parameters: ~0.37M (0.358M - 0.37M)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


class InitialBlock(nn.Module):
    """Initial downsampling block: parallel conv and maxpool concatenated."""
    def __init__(self, in_channels: int = 3, out_channels: int = 16):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels - in_channels, kernel_size=3, stride=2, padding=1, bias=False)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(out_channels)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat([self.conv(x), self.pool(x)], dim=1)
        out = self.bn(out)
        return self.prelu(out)


class Bottleneck(nn.Module):
    """ENet bottleneck module supporting regular, downsampling, dilated, and asymmetric convs."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        internal_ratio: int = 4,
        downsampling: bool = False,
        dilated: int = 1,
        asymmetric: int = 0,
        dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.downsampling = downsampling
        internal_channels = in_channels // internal_ratio

        # Main branch
        if downsampling:
            self.maxpool = nn.MaxPool2d(2, stride=2)
            self.conv_main = nn.Sequential(
                nn.Conv2d(in_channels, internal_channels, kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels)
            )
        else:
            self.conv_main = nn.Sequential(
                nn.Conv2d(in_channels, internal_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels)
            )

        if asymmetric > 0:
            self.conv_mid = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(asymmetric, 1), padding=(asymmetric // 2, 0), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels),
                nn.Conv2d(internal_channels, internal_channels, kernel_size=(1, asymmetric), padding=(0, asymmetric // 2), bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels)
            )
        elif dilated > 1:
            self.conv_mid = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=dilated, dilation=dilated, bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels)
            )
        else:
            self.conv_mid = nn.Sequential(
                nn.Conv2d(internal_channels, internal_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(internal_channels),
                nn.PReLU(internal_channels)
            )

        self.conv_end = nn.Sequential(
            nn.Conv2d(internal_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.prelu = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor):
        if self.downsampling:
            shortcut = F.max_pool2d(x, kernel_size=2, stride=2)
            extra_channels = self.conv_end[0].out_channels - x.size(1)
            if extra_channels > 0:
                pad = torch.zeros(shortcut.size(0), extra_channels, shortcut.size(2), shortcut.size(3), device=x.device)
                shortcut = torch.cat([shortcut, pad], dim=1)
        else:
            shortcut = x

        out = self.conv_main(x)
        out = self.conv_mid(out)
        out = self.conv_end(out)
        out = self.dropout(out)

        out = self.prelu(out + shortcut)
        return out


@register_model('enet')
class ENet(nn.Module):
    """Efficient Neural Network (ENet) for fast real-time semantic segmentation (~0.37M params)."""
    def __init__(self, num_classes: int = 1, in_channels: int = 3):
        super().__init__()
        self.initial = InitialBlock(in_channels, 16)

        # Stage 1
        self.b10 = Bottleneck(16, 64, downsampling=True, dropout_prob=0.01)
        self.b11 = Bottleneck(64, 64, dropout_prob=0.01)
        self.b12 = Bottleneck(64, 64, dropout_prob=0.01)
        self.b13 = Bottleneck(64, 64, dropout_prob=0.01)
        self.b14 = Bottleneck(64, 64, dropout_prob=0.01)

        # Stage 2
        self.b20 = Bottleneck(64, 128, downsampling=True)
        self.b21 = Bottleneck(128, 128)
        self.b22 = Bottleneck(128, 128, dilated=2)
        self.b23 = Bottleneck(128, 128, asymmetric=5)
        self.b24 = Bottleneck(128, 128, dilated=4)
        self.b25 = Bottleneck(128, 128)
        self.b26 = Bottleneck(128, 128, dilated=8)
        self.b27 = Bottleneck(128, 128, asymmetric=5)
        self.b28 = Bottleneck(128, 128, dilated=16)

        # Stage 3
        self.b31 = Bottleneck(128, 128)
        self.b32 = Bottleneck(128, 128, dilated=2)
        self.b33 = Bottleneck(128, 128, asymmetric=5)
        self.b34 = Bottleneck(128, 128, dilated=4)
        self.b35 = Bottleneck(128, 128)
        self.b36 = Bottleneck(128, 128, dilated=8)
        self.b37 = Bottleneck(128, 128, asymmetric=5)
        self.b38 = Bottleneck(128, 128, dilated=16)

        # Decoder Stage 4 & 5
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec4 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec5 = nn.Sequential(
            nn.Conv2d(64, 16, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.PReLU(16)
        )

        # Full resolution head
        self.classifier = nn.Conv2d(16, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]

        out = self.initial(x)

        # Stage 1
        out = self.b10(out)
        out = self.b11(out)
        out = self.b12(out)
        out = self.b13(out)
        out = self.b14(out)

        # Stage 2
        out = self.b20(out)
        out = self.b21(out)
        out = self.b22(out)
        out = self.b23(out)
        out = self.b24(out)
        out = self.b25(out)
        out = self.b26(out)
        out = self.b27(out)
        out = self.b28(out)

        # Stage 3
        out = self.b31(out)
        out = self.b32(out)
        out = self.b33(out)
        out = self.b34(out)
        out = self.b35(out)
        out = self.b36(out)
        out = self.b37(out)
        out = self.b38(out)

        # Decoder
        out = self.dec4(self.upsample4(out))
        out = self.dec5(self.upsample5(out))

        logits = self.classifier(out)
        if logits.shape[2:] != orig_size:
            logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
        return logits
