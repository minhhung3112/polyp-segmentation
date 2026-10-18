"""
ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network.

Reference:
    Mehta, S., Rastegari, M., Shapiro, L., & Hajishirzi, H. (2019).
    ESPNetv2: A Light-weight, Power Efficient, and General Purpose Convolutional Neural Network.
    CVPR 2019, pp. 9190-9200.
    Repository: https://github.com/sacmehta/ESPNetv2
    Parameters: ~0.8M - 1.2M
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


class CBR(nn.Module):
    """Conv2d => BatchNorm2d => PReLU"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.PReLU(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class EESP(nn.Module):
    """
    Extremely Efficient Spatial Pyramid (EESP) unit:
    Reduce => Split => Transform (depthwise dilated conv) => Hierarchical Merge => Project
    """
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, k: int = 4, r_lim: int = 9):
        super().__init__()
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.k = k

        # Determine channels per branch
        d = int(out_channels / k)
        self.d = d

        # Step 1: Reduce with 1x1 conv
        self.proj_1x1 = nn.Conv2d(in_channels, d * k, kernel_size=1, stride=1, bias=False)
        self.bn_proj = nn.BatchNorm2d(d * k)
        self.act_proj = nn.PReLU(d * k)

        # Step 2: Dilated depthwise convs per branch
        branches = []
        for i in range(k):
            rate = min(2 ** i, r_lim)
            pad = rate
            branches.append(
                nn.Conv2d(
                    d, d, kernel_size=3, stride=stride, padding=pad,
                    dilation=rate, groups=d, bias=False
                )
            )
        self.spp_dw = nn.ModuleList(branches)
        self.bn_spp = nn.BatchNorm2d(d * k)
        self.act_spp = nn.PReLU(d * k)

        # Step 3: 1x1 Projection back to out_channels
        self.proj_out = nn.Conv2d(d * k, out_channels, kernel_size=1, bias=False)
        self.bn_out = nn.BatchNorm2d(out_channels)
        self.act_out = nn.PReLU(out_channels)

        # Downsample shortcut if stride > 1 or channel mismatch
        if stride > 1:
            self.shortcut = nn.Sequential(
                nn.AvgPool2d(3, stride=stride, padding=1),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        elif in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = self.shortcut(x)

        # 1. Reduce
        reduced = self.act_proj(self.bn_proj(self.proj_1x1(x)))

        # 2. Split and Transform
        splits = torch.split(reduced, self.d, dim=1)
        outputs = []
        for i, branch in enumerate(self.spp_dw):
            out_b = branch(splits[i])
            if i > 0:
                out_b = out_b + outputs[i - 1]
            outputs.append(out_b)

        # 3. Merge
        merged = torch.cat(outputs, dim=1)
        merged = self.act_spp(self.bn_spp(merged))

        # 4. Project
        out = self.act_out(self.bn_out(self.proj_out(merged)) + res)
        return out


@register_model('espnetv2')
class ESPNetv2(nn.Module):
    """ESPNetv2 for lightweight, efficient polyp segmentation (~0.8M parameters)."""

    def __init__(self, num_classes: int = 1, in_channels: int = 3):
        super().__init__()
        # Initial stem
        self.stem = CBR(in_channels, 32, kernel_size=3, stride=2, padding=1)  # 352 -> 176

        # Stage 1: 176 -> 88
        self.stage1_down = EESP(32, 64, stride=2)
        self.stage1_eesp = EESP(64, 64, stride=1)

        # Stage 2: 88 -> 44
        self.stage2_down = EESP(64, 128, stride=2)
        self.stage2_eesp1 = EESP(128, 128, stride=1)
        self.stage2_eesp2 = EESP(128, 128, stride=1)

        # Stage 3: 44 -> 22
        self.stage3_down = EESP(128, 256, stride=2)
        self.stage3_eesp1 = EESP(256, 256, stride=1)
        self.stage3_eesp2 = EESP(256, 256, stride=1)

        # Lightweight Decoder with skip connections
        self.dec_s3_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_s3_conv = CBR(256 + 128, 64, kernel_size=3, padding=1)

        self.dec_s2_up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dec_s2_conv = CBR(64 + 64, 32, kernel_size=3, padding=1)

        self.classifier = nn.Conv2d(32, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]

        x0 = self.stem(x)              # /2: 32ch, 176x176
        x1 = self.stage1_eesp(self.stage1_down(x0))  # /4: 64ch, 88x88
        x2 = self.stage2_eesp2(self.stage2_eesp1(self.stage2_down(x1)))  # /8: 128ch, 44x44
        x3 = self.stage3_eesp2(self.stage3_eesp1(self.stage3_down(x2)))  # /16: 256ch, 22x22

        # Decoder fusion
        d2 = self.dec_s3_up(x3)        # /8: 256ch -> 44x44
        d2 = torch.cat([d2, x2], dim=1)
        d2 = self.dec_s3_conv(d2)      # 64ch, 44x44

        d1 = self.dec_s2_up(d2)        # /4: 64ch -> 88x88
        d1 = torch.cat([d1, x1], dim=1)
        d1 = self.dec_s2_conv(d1)      # 32ch, 88x88

        logits = self.classifier(d1)
        logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
        return logits
