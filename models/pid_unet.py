"""
Tiny PID-UNet: Ultra-Lightweight Boundary-Attention-Guided U-Net (< 1M Parameters).

Target:
- Strictly < 1.0M parameters (Only ~0.56M params, ~2.2 MB weights).
- Ultra-fast training & real-time inference suitable for Edge AI / Endoscopic video (60+ FPS).
- Retains core U-Net symmetric topology while infusing PID Control & BAG mechanisms:
    1. Proportional (P): Multi-scale spatial details via lightweight skip connections.
    2. Integral (I): Global context aggregation via Light Pyramid Pooling (PPM).
    3. Derivative (D): High-frequency boundary stream producing a 1-channel spatial boundary map.
    4. BAG (Boundary-Attention Guidance): Dynamic boundary-guided fusion of P and I.
    5. Depthwise Separable Convolutions (DW-Conv + PW-Conv): Compresses parameters by ~8x.
    6. Deep Supervision: 4 multi-scale lateral prediction heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# Ultra-Lightweight Building Blocks (Depthwise Separable Convolutions)
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv(nn.Module):
    """
    MobileNet-style Depthwise Separable Convolution (DW 3x3 + PW 1x1).
    Reduces parameter count and computational complexity by ~8-9x compared to standard 3x3 Conv.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.dw = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, dilation=dilation,
                            groups=in_channels, bias=bias)
        self.pw = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.pw(self.dw(x))))


class LightDoubleConv(nn.Module):
    """
    Lightweight residual double-convolution block with depthwise separable layers.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)
        self.shortcut = (
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            ) if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv2(self.conv1(x)) + self.shortcut(x)


# ---------------------------------------------------------------------------
# Integral Component: Light Context PPM (Pyramid Pooling Module)
# ---------------------------------------------------------------------------

class LightPPM(nn.Module):
    """
    Integral (I) Component:
    Aggregates multi-scale global context across large receptive fields using
    adaptive pooling and depthwise separable fusion (< 0.05M params).
    """
    def __init__(self, in_channels: int, out_channels: int, bins=(1, 2, 4)):
        super().__init__()
        reduction_c = max(16, in_channels // 4)
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, reduction_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_c),
                nn.ReLU(inplace=True)
            )
            for bin_size in bins
        ])
        mid_channels = in_channels + reduction_c * len(bins)
        self.fuse = DepthwiseSeparableConv(mid_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        priors = [x]
        for stage in self.stages:
            priors.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True))
        out = torch.cat(priors, dim=1)
        return self.fuse(out)


# ---------------------------------------------------------------------------
# Derivative Component: Lightweight Boundary Stream (D-Branch)
# ---------------------------------------------------------------------------

class LightBoundaryBranch(nn.Module):
    """
    Derivative (D) Component:
    Extracts high-frequency boundary transitions and outputs a 1-channel spatial edge map.
    """
    def __init__(self, c1: int, c2: int, c3: int, d_channels: int = 32):
        super().__init__()
        self.red1 = nn.Conv2d(c1, d_channels, kernel_size=1, bias=False)
        self.red2 = nn.Conv2d(c2, d_channels, kernel_size=1, bias=False)
        self.red3 = nn.Conv2d(c3, d_channels, kernel_size=1, bias=False)
        self.d_fuse = nn.Sequential(
            DepthwiseSeparableConv(d_channels * 3, d_channels, kernel_size=3, padding=1),
            nn.Conv2d(d_channels, 1, kernel_size=1)  # 1-channel boundary logits
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor) -> torch.Tensor:
        target_size = e2.shape[2:]  # H/2, W/2
        r1 = F.interpolate(self.red1(e1), size=target_size, mode='bilinear', align_corners=True)
        r2 = self.red2(e2)
        r3 = F.interpolate(self.red3(e3), size=target_size, mode='bilinear', align_corners=True)
        boundary_map = self.d_fuse(torch.cat([r1, r2, r3], dim=1))
        return boundary_map


# ---------------------------------------------------------------------------
# Light-BAG: Lightweight Boundary-Attention-Guided Fusion Block
# ---------------------------------------------------------------------------

class LightBAG(nn.Module):
    """
    Light-BAG Module:
    Selectively fuses Proportional detail (P) and Integral context (I) guided by
    the 1-channel Derivative boundary attention map A in [0, 1].
    """
    def __init__(self, p_channels: int, i_channels: int, out_channels: int):
        super().__init__()
        self.proj_p = nn.Sequential(
            nn.Conv2d(p_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.proj_i = nn.Sequential(
            nn.Conv2d(i_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.fuse = DepthwiseSeparableConv(out_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, p_feat: torch.Tensor, i_feat: torch.Tensor, boundary_map: torch.Tensor) -> torch.Tensor:
        h, w = p_feat.shape[2:]
        if i_feat.shape[2:] != (h, w):
            i_feat = F.interpolate(i_feat, size=(h, w), mode='bilinear', align_corners=True)
        if boundary_map.shape[2:] != (h, w):
            boundary_map = F.interpolate(boundary_map, size=(h, w), mode='bilinear', align_corners=True)

        p = self.proj_p(p_feat)
        i = self.proj_i(i_feat)
        att = torch.sigmoid(boundary_map)  # (B, 1, H, W) broadcasts across channels

        guided_feat = att * p + (1.0 - att) * i
        out = self.fuse(guided_feat) + i
        return out


# ---------------------------------------------------------------------------
# Full Network: Ultra-Lightweight PID-UNet (< 1M Parameters)
# ---------------------------------------------------------------------------

@register_model('pid_unet')
class TinyPIDUNet(nn.Module):
    """
    Ultra-Lightweight PID-UNet for Polyp Segmentation (< 1M Parameters).
    Total Parameters: ~0.564M (563,957 parameters) - ~55x smaller than canonical U-Net!
    
    Structure:
    - Base channels: [48, 96, 144, 192, 288]
    - Lightweight Depthwise Separable Convolutions throughout.
    - Integral Bottleneck: Multi-scale LightPPM.
    - Derivative Stream: 1-channel boundary attention guidance.
    - 4-Scale Deep Supervision for stable multi-scale learning.
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_c: int = 48):
        super().__init__()
        c1 = base_c          # 48
        c2 = base_c * 2      # 96
        c3 = base_c * 3      # 144
        c4 = base_c * 4      # 192
        c5 = base_c * 6      # 288

        # ----------------- Stem & Encoder -----------------
        self.inc = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True)
        )
        self.enc1 = LightDoubleConv(c1, c1)
        self.down1 = nn.MaxPool2d(2, 2)

        self.enc2 = LightDoubleConv(c1, c2)
        self.down2 = nn.MaxPool2d(2, 2)

        self.enc3 = LightDoubleConv(c2, c3)
        self.down3 = nn.MaxPool2d(2, 2)

        self.enc4 = LightDoubleConv(c3, c4)
        self.down4 = nn.MaxPool2d(2, 2)

        # ----------------- Bottleneck (Integral Context: LightPPM) -----------------
        self.bottleneck = LightPPM(c4, c5, bins=(1, 2, 4))

        # ----------------- Derivative Boundary Branch -----------------
        self.boundary_branch = LightBoundaryBranch(c1, c2, c3, d_channels=32)

        # ----------------- Decoder with Light-BAG Modules -----------------
        self.bag4 = LightBAG(p_channels=c4, i_channels=c5, out_channels=c4)
        self.bag3 = LightBAG(p_channels=c3, i_channels=c4, out_channels=c3)
        self.bag2 = LightBAG(p_channels=c2, i_channels=c3, out_channels=c2)
        self.bag1 = LightBAG(p_channels=c1, i_channels=c2, out_channels=c1)

        # ----------------- Deep Supervision Heads -----------------
        self.head4 = nn.Conv2d(c4, n_classes, kernel_size=1)
        self.head3 = nn.Conv2d(c3, n_classes, kernel_size=1)
        self.head2 = nn.Conv2d(c2, n_classes, kernel_size=1)
        self.head1 = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        sz = x.shape[2:]

        # 1. Encoder Forward
        x0 = self.inc(x)
        e1 = self.enc1(x0)             # (B, 48, H, W)
        e2 = self.enc2(self.down1(e1)) # (B, 96, H/2, W/2)
        e3 = self.enc3(self.down2(e2)) # (B, 144, H/4, W/4)
        e4 = self.enc4(self.down3(e3)) # (B, 192, H/8, W/8)

        # 2. Integral Context Bottleneck
        i_feat = self.bottleneck(self.down4(e4)) # (B, 288, H/16, W/16)

        # 3. Derivative Boundary Stream
        b_map = self.boundary_branch(e1, e2, e3) # (B, 1, H/2, W/2)

        # 4. Decoder with Light-BAG
        d4 = self.bag4(e4, i_feat, b_map)  # (B, 192, H/8, W/8)
        d3 = self.bag3(e3, d4, b_map)      # (B, 144, H/4, W/4)
        d2 = self.bag2(e2, d3, b_map)      # (B, 96, H/2, W/2)
        d1 = self.bag1(e1, d2, b_map)      # (B, 48, H, W)

        # 5. Multi-Scale Deep Supervision Heads
        out4 = F.interpolate(self.head4(d4), size=sz, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.head3(d3), size=sz, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(d2), size=sz, mode='bilinear', align_corners=True)
        out1 = self.head1(d1)  # already (H, W)

        return out4, out3, out2, out1
