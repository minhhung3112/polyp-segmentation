"""
EDANet: Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation.

Reference:
    Lo, S. Y., Hang, H. M., Chan, S. W., & Lin, J. J. (2019).
    Efficient Dense Modules of Asymmetric Convolution for Real-Time Semantic Segmentation.
    IEEE Transactions on Multimedia, 21(10), 2548-2560.
    Paper: arXiv:1809.06364 / Official repo: https://github.com/shaoyuanlo/EDANet
    Target Parameter count: ~0.68M parameters.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


class DownsamplerBlock(nn.Module):
    """
    Downsamples spatial resolution by factor of 2 via parallel 3x3 conv and maxpool,
    concatenated and normalized with BatchNorm + ReLU.
    """
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if out_channels > in_channels:
            conv_out = out_channels - in_channels
            self.conv = nn.Conv2d(in_channels, conv_out, kernel_size=3, stride=2, padding=1, bias=False)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool_proj = nn.Identity()
        else:
            conv_out = out_channels // 2
            pool_out = out_channels - conv_out
            self.conv = nn.Conv2d(in_channels, conv_out, kernel_size=3, stride=2, padding=1, bias=False)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool_proj = nn.Conv2d(in_channels, pool_out, kernel_size=1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c = self.conv(x)
        p = self.pool_proj(self.pool(x))
        out = torch.cat([c, p], dim=1)
        return self.relu(self.bn(out))


class EDAModule(nn.Module):
    """
    EDA Module: Asymmetric 3x1 and 1x3 convolutions with dilation rate d
    and dense connection to preserve feature reuse.
    """
    def __init__(self, in_channels: int, growth_rate: int = 40, dilation: int = 1, dropout_prob: float = 0.05):
        super().__init__()
        pad = dilation
        self.conv1 = nn.Conv2d(in_channels, growth_rate, kernel_size=(3, 1), padding=(pad, 0), dilation=(dilation, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(growth_rate)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(growth_rate, growth_rate, kernel_size=(1, 3), padding=(0, pad), dilation=(1, dilation), bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.dropout(out)
        return torch.cat([x, out], dim=1)


class EDABlock(nn.Module):
    """A dense cascade of multiple EDA modules."""
    def __init__(self, in_channels: int, num_modules: int, growth_rate: int = 40, dilations: list = None):
        super().__init__()
        if dilations is None:
            dilations = [1] * num_modules

        modules = []
        cur_ch = in_channels
        for i in range(num_modules):
            d = dilations[i] if i < len(dilations) else 1
            m = EDAModule(cur_ch, growth_rate=growth_rate, dilation=d)
            modules.append(m)
            cur_ch += growth_rate

        self.layers = nn.ModuleList(modules)
        self.out_channels = cur_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for m in self.layers:
            x = m(x)
        return x


@register_model('edanet')
class EDANet(nn.Module):
    """EDANet: Efficient Dense Asymmetric Convolution Network for real-time segmentation (~0.68M params)."""

    def __init__(self, num_classes: int = 1, in_channels: int = 3):
        super().__init__()
        # Initial downsampling stages: 352 -> 176 -> 88
        self.down1 = DownsamplerBlock(in_channels, 16)   # 3 -> 16
        self.down2 = DownsamplerBlock(16, 64)           # 16 -> 64

        # EDA Block 1: 5 modules, growth rate 40, dilations [1, 1, 1, 2, 2]
        self.eda1 = EDABlock(64, num_modules=5, growth_rate=40, dilations=[1, 1, 1, 2, 2])

        # Downsampling stage 3: 88 -> 44 (channels: 264 -> 128)
        self.down3 = DownsamplerBlock(self.eda1.out_channels, 128)

        # EDA Block 2: 8 modules with larger dilations: [2, 2, 4, 4, 8, 8, 14, 14]
        self.eda2 = EDABlock(128, num_modules=8, growth_rate=40, dilations=[2, 2, 4, 4, 8, 8, 14, 14])

        # Classifier projection head
        self.classifier = nn.Sequential(
            nn.Conv2d(self.eda2.out_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_size = x.shape[2:]

        out = self.down1(x)    # /2
        out = self.down2(out)  # /4
        out = self.eda1(out)
        out = self.down3(out)  # /8
        out = self.eda2(out)

        logits = self.classifier(out)
        # Interpolate back to original resolution (352x352)
        logits = F.interpolate(logits, size=orig_size, mode='bilinear', align_corners=False)
        return logits
