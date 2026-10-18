"""
PID-UNet: Boundary-Attention-Guided U-Net inspired by PID Control Theory for Polyp Segmentation.

Inspired by:
1. Ronneberger et al. "U-Net: Convolutional Networks for Biomedical Image Segmentation" (MICCAI 2015).
2. Xu et al. "PIDNet: A Real-time Semantic Segmentation Network Inspired by PID Controller" (CVPR 2023).

Core Philosophy:
- Proportional (P) Component: High-resolution spatial detail features from early encoder skip connections.
- Integral (I) Component: Low-frequency semantic context features with long-range receptive fields (Bottleneck + PPM).
- Derivative (D) Component: High-frequency boundary/edge extraction stream acting as a damping factor against overshoot.
- BAG (Boundary-Attention-Guided) Fusion: Uses the boundary feature to modulate detail (P) and context (I) features:
    F_fuse = Conv( A * F_P + (1 - A) * F_I ) where A = sigmoid(Conv(F_D))
- Deep Supervision: 4 output maps (lat5, lat4, lat3, lat2) for seamless training in the Polyp benchmark framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# Basic Building Blocks
# ---------------------------------------------------------------------------

class ConvBNReLU(nn.Module):
    """Standard (Conv2d -> BatchNorm -> ReLU) block."""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 1, dilation: int = 1, bias: bool = False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                      stride=stride, padding=padding, dilation=dilation, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class DoubleConv(nn.Module):
    """Two consecutive ConvBNReLU layers."""
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNReLU(in_channels, out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


# ---------------------------------------------------------------------------
# Integral Component: Context Aggregation (Pyramid Pooling Module)
# ---------------------------------------------------------------------------

class ContextIntegralModule(nn.Module):
    """
    Integral (I) Component:
    Aggregates multi-scale global context across large receptive fields to eliminate
    steady-state errors and accurately capture camouflaged polyps.
    """
    def __init__(self, in_channels: int, out_channels: int, bins=(1, 2, 4, 8)):
        super().__init__()
        self.stages = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin_size),
                nn.Conv2d(in_channels, in_channels // len(bins), kernel_size=1, bias=False),
                nn.BatchNorm2d(in_channels // len(bins)),
                nn.ReLU(inplace=True)
            )
            for bin_size in bins
        ])
        mid_channels = in_channels + (in_channels // len(bins)) * len(bins)
        self.bottleneck = nn.Sequential(
            ConvBNReLU(mid_channels, out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h, w = x.shape[2:]
        priors = [x]
        for stage in self.stages:
            pooled = stage(x)
            priors.append(F.interpolate(pooled, size=(h, w), mode='bilinear', align_corners=True))
        out = torch.cat(priors, dim=1)
        return self.bottleneck(out)


# ---------------------------------------------------------------------------
# Derivative Component: Boundary / Edge Extraction Stream
# ---------------------------------------------------------------------------

class BoundaryDerivativeBranch(nn.Module):
    """
    Derivative (D) Component:
    Captures high-frequency spatial transitions (boundaries/contours) between lesions
    and normal mucosa. Acts as a damping factor preventing edge overshoot.
    """
    def __init__(self, channels_list: list[int], d_channels: int = 64):
        super().__init__()
        # channels_list: [c1, c2, c3] corresponding to 1x, 1/2x, 1/4x resolutions
        c1, c2, c3 = channels_list
        
        self.reduce1 = ConvBNReLU(c1, d_channels, kernel_size=1, padding=0)
        self.reduce2 = ConvBNReLU(c2, d_channels, kernel_size=1, padding=0)
        self.reduce3 = ConvBNReLU(c3, d_channels, kernel_size=1, padding=0)
        
        # High-pass boundary refinement
        self.fuse = nn.Sequential(
            ConvBNReLU(d_channels * 3, d_channels, kernel_size=3, padding=1),
            ConvBNReLU(d_channels, d_channels, kernel_size=3, padding=1)
        )

    def forward(self, e1: torch.Tensor, e2: torch.Tensor, e3: torch.Tensor) -> torch.Tensor:
        # Align multi-scale shallow features to e2 resolution (H/2, W/2)
        target_size = e2.shape[2:]
        r1 = F.interpolate(self.reduce1(e1), size=target_size, mode='bilinear', align_corners=True)
        r2 = self.reduce2(e2)
        r3 = F.interpolate(self.reduce3(e3), size=target_size, mode='bilinear', align_corners=True)
        
        boundary_feat = self.fuse(torch.cat([r1, r2, r3], dim=1))
        return boundary_feat


# ---------------------------------------------------------------------------
# BAG: Boundary-Attention-Guided Fusion Block
# ---------------------------------------------------------------------------

class BAGModule(nn.Module):
    """
    Boundary-Attention-Guided (BAG) Fusion Module.
    
    Dynamically balances Proportional (Detail, P) and Integral (Context, I) features
    using the Derivative (Boundary, D) attention map:
    - At boundaries (A -> 1): Rely on high-resolution spatial details (P) to sharpen edges.
    - Inside lesions & background (A -> 0): Rely on contextual semantics (I) to suppress noise.
    """
    def __init__(self, p_channels: int, i_channels: int, out_channels: int, d_channels: int = 64):
        super().__init__()
        # Channel adapters
        self.proj_p = ConvBNReLU(p_channels, out_channels, kernel_size=1, padding=0)
        self.proj_i = ConvBNReLU(i_channels, out_channels, kernel_size=1, padding=0)
        
        # Boundary attention gate
        self.gate_conv = nn.Sequential(
            nn.Conv2d(d_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )
        
        # Fused representation refinement with residual shortcut
        self.fuse_conv = nn.Sequential(
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1),
            ConvBNReLU(out_channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, p_feat: torch.Tensor, i_feat: torch.Tensor, d_feat: torch.Tensor) -> torch.Tensor:
        # Align spatial sizes if needed
        h, w = p_feat.shape[2:]
        if i_feat.shape[2:] != (h, w):
            i_feat = F.interpolate(i_feat, size=(h, w), mode='bilinear', align_corners=True)
        if d_feat.shape[2:] != (h, w):
            d_feat = F.interpolate(d_feat, size=(h, w), mode='bilinear', align_corners=True)

        p = self.proj_p(p_feat)
        i = self.proj_i(i_feat)
        att = self.gate_conv(d_feat)  # Boundary Attention map in [0, 1]

        # BAG weighted blending: near boundary emphasize P; interior/background emphasize I
        guided_feat = att * p + (1.0 - att) * i
        
        # Refinement with residual context
        out = self.fuse_conv(guided_feat) + i
        return out


# ---------------------------------------------------------------------------
# Full Network: PID-UNet
# ---------------------------------------------------------------------------

@register_model('pid_unet')
class PIDUNet(nn.Module):
    """
    PID-UNet: An End-to-End Polyp Segmentation Architecture fusing U-Net, PID Control,
    and Boundary-Attention Guidance (BAG).
    
    Key Features:
    - Symmetric U-Net backbone with enhanced multi-scale feature propagation.
    - Triple-branch PID synergy:
        * P-Features: Fine spatial detail skips from encoder stages.
        * I-Features: Integral context aggregator at the bottleneck.
        * D-Features: Dedicated high-frequency boundary extraction stream.
    - BAG fusion units replace plain naive concatenations in the decoder stages.
    - Deep supervision with 4 lateral prediction maps matching benchmark standards.
    """
    def __init__(self, in_channels: int = 3, n_classes: int = 1, base_c: int = 32):
        super().__init__()
        c1, c2, c3, c4, c5 = base_c, base_c * 2, base_c * 4, base_c * 8, base_c * 16
        d_c = 64

        # ----------------- Encoder -----------------
        self.enc1 = DoubleConv(in_channels, c1)       # H, W
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = DoubleConv(c1, c2)                # H/2, W/2
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = DoubleConv(c2, c3)                # H/4, W/4
        self.pool3 = nn.MaxPool2d(2, 2)
        
        self.enc4 = DoubleConv(c3, c4)                # H/8, W/8
        self.pool4 = nn.MaxPool2d(2, 2)
        
        # ----------------- Bottleneck (Integral I-Component) -----------------
        self.bottleneck = ContextIntegralModule(c4, c5)  # H/16, W/16

        # ----------------- Derivative (D-Component: Boundary Branch) ---------
        self.boundary_branch = BoundaryDerivativeBranch([c1, c2, c3], d_channels=d_c)

        # ----------------- Decoder with BAG Modules -----------------
        # Level 4: Fuse enc4 (P) and bottleneck (I)
        self.bag4 = BAGModule(p_channels=c4, i_channels=c5, out_channels=c4, d_channels=d_c)
        
        # Level 3: Fuse enc3 (P) and dec4 (I)
        self.bag3 = BAGModule(p_channels=c3, i_channels=c4, out_channels=c3, d_channels=d_c)
        
        # Level 2: Fuse enc2 (P) and dec3 (I)
        self.bag2 = BAGModule(p_channels=c2, i_channels=c3, out_channels=c2, d_channels=d_c)
        
        # Level 1: Fuse enc1 (P) and dec2 (I)
        self.bag1 = BAGModule(p_channels=c1, i_channels=c2, out_channels=c1, d_channels=d_c)

        # ----------------- Deep Supervision Heads -----------------
        self.head4 = nn.Conv2d(c4, n_classes, kernel_size=1)
        self.head3 = nn.Conv2d(c3, n_classes, kernel_size=1)
        self.head2 = nn.Conv2d(c2, n_classes, kernel_size=1)
        self.head1 = nn.Conv2d(c1, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor):
        input_size = x.shape[2:]

        # 1. Encoder Forward Pass
        e1 = self.enc1(x)            # (B, c1, H, W)
        e2 = self.enc2(self.pool1(e1)) # (B, c2, H/2, W/2)
        e3 = self.enc3(self.pool2(e2)) # (B, c3, H/4, W/4)
        e4 = self.enc4(self.pool3(e3)) # (B, c4, H/8, W/8)

        # 2. Integral (I) Context Bottleneck
        i_feat = self.bottleneck(self.pool4(e4)) # (B, c5, H/16, W/16)

        # 3. Derivative (D) Boundary Branch
        d_feat = self.boundary_branch(e1, e2, e3) # (B, d_c, H/2, W/2)

        # 4. Decoder with Boundary-Attention Guidance (BAG)
        d4 = self.bag4(p_feat=e4, i_feat=i_feat, d_feat=d_feat)  # (B, c4, H/8, W/8)
        d3 = self.bag3(p_feat=e3, i_feat=d4, d_feat=d_feat)      # (B, c3, H/4, W/4)
        d2 = self.bag2(p_feat=e2, i_feat=d3, d_feat=d_feat)      # (B, c2, H/2, W/2)
        d1 = self.bag1(p_feat=e1, i_feat=d2, d_feat=d_feat)      # (B, c1, H, W)

        # 5. Multi-Scale Deep Supervision (PraNet 4-scale convention: lat5, lat4, lat3, lat2)
        # Rescale all heads to match input resolution (H, W)
        out4 = F.interpolate(self.head4(d4), size=input_size, mode='bilinear', align_corners=True)
        out3 = F.interpolate(self.head3(d3), size=input_size, mode='bilinear', align_corners=True)
        out2 = F.interpolate(self.head2(d2), size=input_size, mode='bilinear', align_corners=True)
        out1 = self.head1(d1)  # already (H, W)

        # Order matches standard deep supervision: coarsest to finest (finest is out1 at the end)
        return out4, out3, out2, out1
