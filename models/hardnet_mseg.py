"""
HarDNet-MSEG: A Metric-dimensioned Segmentor for Polyp Segmentation.

Reference:
    Huang, C. H., Wu, H. Y., & Lin, Y. L. (2021).
    Hardnet-mseg: A computing-efficient and accuracy-balanced segmentor for medical image segmentation.
    Computer Methods and Programs in Biomedicine, 208, 106288.

Backbone: HarDNet-68 (Harmonic DenseNet)
Architecture: HarDNet68 backbone + RFB + Cascade Partial Decoder Aggregation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models import register_model


# ---------------------------------------------------------------------------
# HarDNet Building Blocks
# ---------------------------------------------------------------------------

class ConvLayer(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int = 3, stride: int = 1, bias: bool = False):
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_channels, out_channels, kernel_size=kernel,
            stride=stride, padding=kernel // 2, bias=bias
        ))
        self.add_module('norm', nn.BatchNorm2d(out_channels))
        self.add_module('relu', nn.ReLU6(inplace=True))


class HarDBlock(nn.Module):
    def get_link(self, layer, base_ch, growth_rate, grmul):
        if layer == 0:
            return base_ch, 0, []
        out_channels = growth_rate
        link = []
        for i in range(10):
            dv = 2 ** i
            if layer % dv == 0:
                k = layer - dv
                link.append(k)
                if i > 0:
                    out_channels *= grmul
        out_channels = int(int(out_channels + 1) / 2) * 2
        in_channels = 0
        for i in link:
            ch, _, _ = self.get_link(i, base_ch, growth_rate, grmul)
            in_channels += ch
        return out_channels, in_channels, link

    def __init__(self, in_channels: int, growth_rate: int, grmul: float, n_layers: int, keepBase: bool = False):
        super().__init__()
        self.keepBase = keepBase
        self.links = []
        layers_ = []
        self.out_channels = 0
        for i in range(n_layers):
            outch, inch, link = self.get_link(i + 1, in_channels, growth_rate, grmul)
            self.links.append(link)
            layers_.append(ConvLayer(inch, outch))
            if (i % 2 == 0) or (i == n_layers - 1):
                self.out_channels += outch
        self.layers = nn.ModuleList(layers_)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        layers_ = [x]
        for layer in range(len(self.layers)):
            link = self.links[layer]
            tin = [layers_[i] for i in link]
            out = self.layers[layer](torch.cat(tin, 1) if len(tin) > 1 else tin[0])
            layers_.append(out)
        t = len(layers_)
        out_ = [layers_[i] for i in range(t) if (i == 0 and self.keepBase) or (i == t - 1) or (i % 2 == 1)]
        return torch.cat(out_, 1)


class HarDNet68(nn.Module):
    def __init__(self):
        super().__init__()
        first_ch = [32, 64]
        grmul = 1.7
        ch_list = [128, 256, 320, 640, 1024]
        gr = [14, 16, 20, 40, 160]
        n_layers = [8, 16, 16, 16, 4]
        downSamp = [1, 0, 1, 1, 0]

        self.base = nn.ModuleList([
            ConvLayer(in_channels=3, out_channels=first_ch[0], kernel=3, stride=2, bias=False),
            ConvLayer(first_ch[0], first_ch[1], kernel=3),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ])

        ch = first_ch[1]
        for i in range(len(n_layers)):
            blk = HarDBlock(ch, gr[i], grmul, n_layers[i])
            ch = blk.out_channels
            self.base.append(blk)
            self.base.append(ConvLayer(ch, ch_list[i], kernel=1))
            ch = ch_list[i]
            if downSamp[i] == 1:
                self.base.append(nn.MaxPool2d(kernel_size=2, stride=2))

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        out_branch = []
        for i in range(len(self.base)):
            x = self.base[i](x)
            # Tap intermediate multi-scale feature maps: [128, 320, 640, 1024]
            if i in (4, 9, 12, 15):
                out_branch.append(x)
        return out_branch


# ---------------------------------------------------------------------------
# RFB and Aggregation
# ---------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class RFB_modified(nn.Module):
    def __init__(self, in_channel: int, out_channel: int):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(BasicConv2d(in_channel, out_channel, 1))
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            nn.Conv2d(out_channel, out_channel, 3, padding=7, dilation=7, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        self.conv_cat = BasicConv2d(4 * out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), dim=1))
        return self.relu(x_cat + self.conv_res(x))


class aggregation(nn.Module):
    def __init__(self, channel: int = 32):
        super().__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )
        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1)
        x3_2 = self.conv_concat3(x3_2)
        return self.conv5(self.conv4(x3_2))


# ---------------------------------------------------------------------------
# HarDNet-MSEG Network
# ---------------------------------------------------------------------------

@register_model('hardnet_mseg')
class HarDMSEG(nn.Module):
    """HarDNet-MSEG: High-speed, high-efficiency polyp segmentation network."""

    def __init__(self, channel: int = 32):
        super().__init__()
        self.hardnet = HarDNet68()
        self.rfb2_1 = RFB_modified(320, channel)
        self.rfb3_1 = RFB_modified(640, channel)
        self.rfb4_1 = RFB_modified(1024, channel)
        self.agg1 = aggregation(channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hardnetout = self.hardnet(x)
        # hardnetout channels: [256 (stride 4), 320 (stride 8), 640 (stride 16), 1024 (stride 32)]
        x2 = hardnetout[1]   # 320 ch, 44x44
        x3 = hardnetout[2]   # 640 ch, 22x22
        x4 = hardnetout[3]   # 1024 ch, 11x11

        x2_rfb = self.rfb2_1(x2)
        x3_rfb = self.rfb3_1(x3)
        x4_rfb = self.rfb4_1(x4)

        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)
        lateral_map = F.interpolate(ra5_feat, scale_factor=8, mode='bilinear', align_corners=False)
        return lateral_map
