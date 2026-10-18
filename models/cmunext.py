"""
CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion.

Reference:
    Tang, F., Wang, Q., Jiang, L., Zhou, Q., & Li, K. (2024).
    CMUNeXt: An Efficient Medical Image Segmentation Network based on
    Large Kernel and Skip Fusion.
    arXiv:2308.01239
    https://github.com/FengheTan9/CMUNeXt

Architecture (faithfully from network/CMUNeXt.py):
    CMUNeXtBlock: depthwise square-kernel conv (k×k) + inverted bottleneck + residual
    Encoder: stem + 4 downsampling CMUNeXtBlocks
    Decoder: 4 upsampling stages with skip connections
    Default: base variant (smallest) — channels=[16,32,64,128,256], depths=[2,2,2,2], k=3

Variants (from main.py):
    cmunext   (base, default): channels=[16,32,64,128,256], depths=[2,2,2,2], k=3  → ~0.72M
    cmunext_s (small):         channels=[32,64,128,256,512], depths=[2,2,2,2], k=5  → ~9.3M
    cmunext_l (large):         channels=[64,128,256,512,1024],depths=[1,1,1,1],k=7  → ~34.2M

We register the base (smallest) variant.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# Building blocks (faithful to https://github.com/FengheTan9/CMUNeXt)
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    """
    CMUNeXt encoder block.
    Uses square depthwise conv (k×k) exactly as in the official repo:
        nn.Conv2d(ch_in, ch_in, kernel_size=(k,k), groups=ch_in, padding=(k//2, k//2))
    followed by inverted bottleneck (×4 expand) + residual.
    """

    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super().__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in,
                              padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(ch_in),
            ) for _ in range(depth)]
        )
        self.up = ConvBlock(ch_in, ch_out)

    def forward(self, x):
        return self.up(self.block(x))


class ConvBlock(nn.Module):
    """3×3 Conv → BN → ReLU."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpConv(nn.Module):
    """2× bilinear upsample + ConvBlock."""

    def __init__(self, ch_in, ch_out):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            ConvBlock(ch_in, ch_out),
        )

    def forward(self, x):
        return self.up(x)


# ---------------------------------------------------------------------------
# CMUNeXt (base / smallest variant)
# ---------------------------------------------------------------------------

@register_model('cmunext')
class CMUNeXt(nn.Module):
    """
    CMUNeXt base (smallest) variant.
    channels=[16,32,64,128,256], depths=[2,2,2,2], k=3
    """

    def __init__(
        self,
        img_ch: int = 3,
        output_ch: int = 1,
        channels=None,
        depths=None,
        k: int = 3,
    ):
        super().__init__()
        if channels is None:
            channels = [16, 32, 64, 128, 256]   # base variant (smallest)
        if depths is None:
            depths = [2, 2, 2, 2]

        # Stem
        self.stem = ConvBlock(img_ch, channels[0])

        # Encoder (4 stages with max pooling between)
        self.enc1 = CMUNeXtBlock(channels[0], channels[1], depth=depths[0], k=k)
        self.enc2 = CMUNeXtBlock(channels[1], channels[2], depth=depths[1], k=k)
        self.enc3 = CMUNeXtBlock(channels[2], channels[3], depth=depths[2], k=k)
        self.enc4 = CMUNeXtBlock(channels[3], channels[4], depth=depths[3], k=k)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(channels[4], channels[4])

        # Decoder (skip cat: up_out + enc_skip → dec)
        self.up4 = UpConv(channels[4], channels[3])
        self.dec4 = ConvBlock(channels[3] + channels[4], channels[3])

        self.up3 = UpConv(channels[3], channels[2])
        self.dec3 = ConvBlock(channels[2] + channels[3], channels[2])

        self.up2 = UpConv(channels[2], channels[1])
        self.dec2 = ConvBlock(channels[1] + channels[2], channels[1])

        self.up1 = UpConv(channels[1], channels[0])
        self.dec1 = ConvBlock(channels[0] + channels[1], channels[0])

        # Output head
        self.out_conv = nn.Conv2d(channels[0], output_ch, kernel_size=1)

    def forward(self, x):
        # Encoder
        s0 = self.stem(x)
        s1 = self.enc1(s0)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        s4 = self.enc4(self.pool(s3))

        # Bottleneck
        b = self.bottleneck(self.pool(s4))

        # Decoder with skip connections
        d4 = self.dec4(torch.cat([self.up4(b),  s4], dim=1))
        d3 = self.dec3(torch.cat([self.up3(d4), s3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), s2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), s1], dim=1))

        return self.out_conv(d1)
