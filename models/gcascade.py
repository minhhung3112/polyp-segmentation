"""
G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation.

Reference:
    Rahman, M.M., & Marculescu, R. (2024).
    G-CASCADE: Efficient Cascaded Graph Convolutional Decoding for 2D Medical Image Segmentation.
    WACV 2024.
    https://github.com/SLDGroup/G-CASCADE

Architecture:
    Encoder : PVT-v2-B2 (via timm)  → 4 feature maps [64, 128, 320, 512]
    Decoder : GCASCADE — Cascaded Upsampling with Graph Convolutional Blocks (GCB)
              GCB uses graph-convolution-style nearest-neighbor aggregation approximated
              via depthwise conv + pointwise conv (linear GCN equivalent on grid graphs).
    Output  : single segmentation map (B, 1, H, W)

Note on GCB: The original uses torch_geometric's Grapher module. We implement an
equivalent Grid-GCN using depthwise separable convolution, which is mathematically
equivalent on regular grid graphs (each pixel's neighbors = local convolution receptive field).

Registered as: 'gcascade'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models import register_model


# ---------------------------------------------------------------------------
# Grid Graph Convolution Block (GCB equivalent without torch_geometric)
# ---------------------------------------------------------------------------

class GridGCB(nn.Module):
    """
    Grid Graph Convolutional Block.
    Approximates Grapher (GCN on grid) via depthwise separable conv.
    On a regular grid, nearest-neighbor aggregation = depthwise conv with local kernel.
    """

    def __init__(self, in_ch, k=9, dilation=1, act='gelu'):
        super().__init__()
        pad = (k // 2) * dilation
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
        )
        # Depthwise conv = graph aggregation over k×k neighborhood
        self.graph_conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, k, padding=pad, dilation=dilation, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.GELU() if act == 'gelu' else nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, in_ch, 1, bias=False),
            nn.BatchNorm2d(in_ch),
        )
        self.fc2 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
        )
        self.act = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.graph_conv(x)
        x = self.fc2(x)
        return self.act(x + shortcut)


class FFN(nn.Module):
    """Feed-Forward Network (pointwise MLP)."""

    def __init__(self, in_ch, hidden_ch=None):
        super().__init__()
        hidden_ch = hidden_ch or in_ch * 4
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden_ch, 1),
            nn.BatchNorm2d(hidden_ch),
            nn.GELU(),
            nn.Conv2d(hidden_ch, in_ch, 1),
            nn.BatchNorm2d(in_ch),
        )

    def forward(self, x):
        return x + self.net(x)


# ---------------------------------------------------------------------------
# Conv Upsampling Block (UCB equivalent)
# ---------------------------------------------------------------------------

class UCB(nn.Module):
    """Upsample + Conv Block."""

    def __init__(self, ch_in, ch_out, activation='gelu'):
        super().__init__()
        act = nn.GELU() if activation == 'gelu' else nn.ReLU(inplace=True)
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(ch_in, ch_in, 3, padding=1, groups=ch_in, bias=True),
            nn.BatchNorm2d(ch_in),
            act,
            nn.Conv2d(ch_in, ch_out, 1, bias=True),
        )

    def forward(self, x):
        return self.up(x)


# ---------------------------------------------------------------------------
# G-CASCADE Decoder
# ---------------------------------------------------------------------------

class GCASCADEDecoder(nn.Module):
    """
    G-CASCADE decoder: cascaded upsampling with GCBs and skip connections.
    channels = [c1, c2, c3, c4] (low→high resolution encoder features)
    """

    def __init__(self, channels, num_classes=1):
        super().__init__()
        c1, c2, c3, c4 = channels

        # Stage 4→3
        self.gcb4 = nn.Sequential(GridGCB(c4), FFN(c4))
        self.up4 = UCB(c4, c3)
        self.proj3 = nn.Conv2d(c3 * 2, c3, 1)

        # Stage 3→2
        self.gcb3 = nn.Sequential(GridGCB(c3), FFN(c3))
        self.up3 = UCB(c3, c2)
        self.proj2 = nn.Conv2d(c2 * 2, c2, 1)

        # Stage 2→1
        self.gcb2 = nn.Sequential(GridGCB(c2), FFN(c2))
        self.up2 = UCB(c2, c1)
        self.proj1 = nn.Conv2d(c1 * 2, c1, 1)

        # Stage 1 refinement
        self.gcb1 = nn.Sequential(GridGCB(c1), FFN(c1))

        # Output
        self.out = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(c1, num_classes, 1),
        )

    def forward(self, x1, x2, x3, x4):
        # x1: stride 4 (H/4), x4: stride 32 (H/32)
        d4 = self.gcb4(x4)
        d3 = self.proj3(torch.cat([self.up4(d4), x3], dim=1))

        d3 = self.gcb3(d3)
        d2 = self.proj2(torch.cat([self.up3(d3), x2], dim=1))

        d2 = self.gcb2(d2)
        d1 = self.proj1(torch.cat([self.up2(d2), x1], dim=1))

        d1 = self.gcb1(d1)
        return self.out(d1)


# ---------------------------------------------------------------------------
# G-CASCADE Net
# ---------------------------------------------------------------------------

@register_model('gcascade')
class GCASCADENet(nn.Module):
    """
    G-CASCADE (WACV 2024): Graph Convolutional Cascade Decoding.

    Backbone: PVT-v2-B0 (smallest variant, via timm, pretrained=True)
              Channels: [32, 64, 160, 256]
    Decoder : G-CASCADE with GridGCB (grid-graph conv equivalent)
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        self.backbone = timm.create_model(
            'pvt_v2_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = [32, 64, 160, 256]
        self.decoder = GCASCADEDecoder(channels=channels, num_classes=num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        return self.decoder(*feats)
