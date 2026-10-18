"""
EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation.

Reference:
    Rahman, M.M., Munir, M., & Marculescu, R. (2024).
    EMCAD: Efficient Multi-scale Convolutional Attention Decoding for Medical Image Segmentation.
    CVPR 2024.
    https://github.com/SLDGroup/EMCAD

Architecture:
    Encoder : PVT-v2-B2 (via timm)  → 4 feature maps [64, 128, 320, 512]
    Decoder : EMCAD decoder with MSDC (Multi-Scale Depth-wise Conv) + LGAG (Local-Global Attention Gate)
    Output  : single segmentation map (B, 1, H, W)

Registered as: 'emcad'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models import register_model


# ---------------------------------------------------------------------------
# EMCAD Decoder Components (from lib/decoders.py)
# ---------------------------------------------------------------------------

def channel_shuffle(x, groups):
    B, C, H, W = x.shape
    cpg = C // groups
    x = x.view(B, groups, cpg, H, W)
    x = x.transpose(1, 2).contiguous()
    return x.view(B, C, H, W)


class MSDC(nn.Module):
    """Multi-Scale Depth-wise Convolution."""

    def __init__(self, in_ch, kernel_sizes=(1, 3, 5), activation='relu6', dw_parallel=True):
        super().__init__()
        self.dw_parallel = dw_parallel
        self.dwconvs = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_ch, in_ch, k, padding=k // 2, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.ReLU6(inplace=True) if activation == 'relu6' else nn.ReLU(inplace=True),
            ) for k in kernel_sizes
        ])

    def forward(self, x):
        if self.dw_parallel:
            return sum(conv(x) for conv in self.dwconvs)
        out = x
        for conv in self.dwconvs:
            out = conv(out)
        return out


class LGAG(nn.Module):
    """Local-Global Attention Gate."""

    def __init__(self, F_g, F_l, F_int, kernel_size=3, groups=1, activation='relu'):
        super().__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, 1, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size, padding=kernel_size // 2, groups=groups, bias=True),
            nn.BatchNorm2d(F_int),
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, 1, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )
        self.act = nn.ReLU(inplace=True) if activation == 'relu' else nn.GELU()

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        if g1.shape[2:] != x1.shape[2:]:
            g1 = F.interpolate(g1, size=x1.shape[2:], mode='bilinear', align_corners=False)
        psi = self.act(g1 + x1)
        psi = self.psi(psi)
        return x * psi


class MBConv(nn.Module):
    """Mobile Inverted Bottleneck with MSDC."""

    def __init__(self, in_ch, out_ch, kernel_sizes=(1, 3, 5), expansion=2, activation='relu6', dw_parallel=True, add=True):
        super().__init__()
        mid = in_ch * expansion
        self.add = add
        self.pw1 = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, bias=False),
            nn.BatchNorm2d(mid),
            nn.ReLU6(inplace=True) if activation == 'relu6' else nn.ReLU(inplace=True),
        )
        self.msdc = MSDC(mid, kernel_sizes, activation, dw_parallel)
        n = len(kernel_sizes)
        self.shuffle = lambda x: channel_shuffle(x, n)
        self.pw2 = nn.Sequential(
            nn.Conv2d(mid, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.act = nn.ReLU6(inplace=True) if activation == 'relu6' else nn.ReLU(inplace=True)
        if in_ch != out_ch:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch))
        else:
            self.skip = None

    def forward(self, x):
        residual = x if self.skip is None else self.skip(x)
        out = self.pw1(x)
        out = self.msdc(out)
        out = self.pw2(out)
        if self.add:
            out = self.act(out + residual)
        else:
            out = self.act(out)
        return out


class EMCADDecoder(nn.Module):
    """EMCAD Decoder: 4 levels of LGAG-gated upsampling + MBConv."""

    def __init__(self, channels, num_classes=1, kernel_sizes=(1, 3, 5), expansion=2,
                 dw_parallel=True, add=True, lgag_ks=3, activation='relu'):
        super().__init__()
        # channels = [c1, c2, c3, c4] from encoder (low→high resolution)
        c1, c2, c3, c4 = channels  # e.g. [64, 128, 320, 512]

        # Decoder top-down
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lgag3 = LGAG(c4, c3, c3 // 2, kernel_size=lgag_ks, activation=activation)
        self.mbconv3 = MBConv(c4 + c3, c3, kernel_sizes, expansion, activation, dw_parallel, add)

        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lgag2 = LGAG(c3, c2, c2 // 2, kernel_size=lgag_ks, activation=activation)
        self.mbconv2 = MBConv(c3 + c2, c2, kernel_sizes, expansion, activation, dw_parallel, add)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.lgag1 = LGAG(c2, c1, c1 // 2, kernel_size=lgag_ks, activation=activation)
        self.mbconv1 = MBConv(c2 + c1, c1, kernel_sizes, expansion, activation, dw_parallel, add)

        # Output head
        self.out_head = nn.Sequential(
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=False),
            nn.Conv2d(c1, num_classes, 1),
        )

    def forward(self, x1, x2, x3, x4):
        # x1: highest res (H/4), x4: lowest res (H/32)
        d3 = self.up3(x4)
        g3 = self.lgag3(d3, x3)
        d3 = self.mbconv3(torch.cat([d3, g3], dim=1))

        d2 = self.up2(d3)
        g2 = self.lgag2(d2, x2)
        d2 = self.mbconv2(torch.cat([d2, g2], dim=1))

        d1 = self.up1(d2)
        g1 = self.lgag1(d1, x1)
        d1 = self.mbconv1(torch.cat([d1, g1], dim=1))

        return self.out_head(d1)


# ---------------------------------------------------------------------------
# EMCAD Net
# ---------------------------------------------------------------------------

@register_model('emcad')
class EMCADNet(nn.Module):
    """
    EMCAD: Efficient Multi-scale Convolutional Attention Decoding (CVPR 2024).

    Backbone: PVT-v2-B0 (smallest variant, via timm, pretrained=True)
              Channels: [32, 64, 160, 256]
    Decoder : EMCAD with MSDC [1,3,5] + LGAG attention gates
    """

    def __init__(self, pretrained: bool = True, num_classes: int = 1):
        super().__init__()
        # PVT-v2-B0 (smallest): channels [32, 64, 160, 256]
        self.backbone = timm.create_model(
            'pvt_v2_b0',
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        channels = [32, 64, 160, 256]
        self.decoder = EMCADDecoder(
            channels=channels,
            num_classes=num_classes,
            kernel_sizes=(1, 3, 5),
            expansion=2,
            dw_parallel=True,
            add=True,
            lgag_ks=3,
            activation='relu',
        )

    def forward(self, x):
        feats = self.backbone(x)     # [x1, x2, x3, x4] at strides [4, 8, 16, 32]
        return self.decoder(*feats)
