"""
Res2Net-50 v1b 26w 4s backbone — self-contained implementation.

Reference:
    Gao, S., Cheng, M. M., Zhao, K., Zhang, X. Y., Yang, M. H., & Torr, P. (2021).
    Res2Net: A new multi-scale backbone architecture.
    IEEE Transactions on Pattern Analysis and Machine Intelligence, 43(2), 652-662.

Pretrained weights (ImageNet):
    https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth
"""

import torch
import torch.nn as nn
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = ["res2net50_v1b_26w_4s"]

_PRETRAINED_URLS = {
    "res2net50_v1b_26w_4s": (
        "https://shanghuagao.oss-cn-beijing.aliyuncs.com/res2net/res2net50_v1b_26w_4s-3cf99910.pth"
    ),
}


class Bottle2neck(nn.Module):
    """Res2Net bottleneck building block (v1b variant).

    Args:
        inplanes  : Number of input channels.
        planes    : Number of intermediate channels (before expansion).
        stride    : Stride for 3×3 convolutions.
        downsample: Optional downsampling module for shortcut.
        baseWidth : Base width per Res2Net branch.
        scale     : Number of feature map groups (scale parameter *s*).
        stype     : ``'normal'`` for non-first blocks, ``'stage'`` for the
                    first block of each stage (uses pool instead of 3×3 conv
                    on the residual split).
    """

    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        baseWidth: int = 26,
        scale: int = 4,
        stype: str = "normal",
    ) -> None:
        super().__init__()

        width = int(planes * (baseWidth / 64.0))

        self.conv1 = nn.Conv2d(inplanes, width * scale, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width * scale)

        if scale == 1:
            self.nums = 1
        else:
            self.nums = scale - 1

        if stype == "stage":
            self.pool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        convs, bns = [], []
        for _ in range(self.nums):
            convs.append(
                nn.Conv2d(
                    width, width, kernel_size=3, stride=stride, padding=1, bias=False
                )
            )
            bns.append(nn.BatchNorm2d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv2d(
            width * scale, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stype = stype
        self.scale = scale
        self.width = width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # Split into *scale* subsets of *width* channels
        spx = torch.split(out, self.width, dim=1)
        sp_list = []
        for i, (conv, bn) in enumerate(zip(self.convs, self.bns)):
            if i == 0 or self.stype == "stage":
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = conv(sp)
            sp = self.relu(bn(sp))
            sp_list.append(sp)

        if self.scale != 1 and self.stype == "normal":
            sp_list.append(spx[self.nums])
        elif self.scale != 1 and self.stype == "stage":
            sp_list.append(self.pool(spx[self.nums]))

        out = torch.cat(sp_list, dim=1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    """Res2Net-style ResNet backbone (v1b — stride-2 at 3×3 instead of 1×1)."""

    def __init__(
        self,
        block: type,
        layers: list[int],
        baseWidth: int = 26,
        scale: int = 4,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.baseWidth = baseWidth
        self.scale = scale

        # v1b: 3-conv stem
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # Weight init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        block: type,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(
                    kernel_size=stride,
                    stride=stride,
                    ceil_mode=True,
                    count_include_pad=False,
                ),
                nn.Conv2d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=1,
                    bias=False,
                ),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                downsample=downsample,
                stype="stage",
                baseWidth=self.baseWidth,
                scale=self.scale,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    baseWidth=self.baseWidth,
                    scale=self.scale,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # /4

        x1 = self.layer1(x)   # /4,  256 ch
        x2 = self.layer2(x1)  # /8,  512 ch
        x3 = self.layer3(x2)  # /16, 1024 ch
        x4 = self.layer4(x3)  # /32, 2048 ch
        return [x1, x2, x3, x4]


def res2net50_v1b_26w_4s(pretrained: bool = True) -> ResNet:
    """Res2Net-50 v1b with baseWidth=26 and scale=4.

    Args:
        pretrained: If ``True``, load ImageNet-pretrained weights.

    Returns:
        ``ResNet`` model instance.
    """
    model = ResNet(Bottle2neck, [3, 4, 6, 3], baseWidth=26, scale=4)

    if pretrained:
        url = _PRETRAINED_URLS["res2net50_v1b_26w_4s"]
        print(f"[Backbone] Loading pretrained Res2Net-50 v1b from:\n  {url}")
        state_dict = load_state_dict_from_url(url, progress=True, map_location="cpu")
        # The downloaded checkpoint uses 'conv1' as a plain Conv2d.
        # Our stem is a Sequential — remap keys gracefully.
        model_sd = model.state_dict()
        new_sd = {}
        for k, v in state_dict.items():
            if k in model_sd and model_sd[k].shape == v.shape:
                new_sd[k] = v
        missing = set(model_sd.keys()) - set(new_sd.keys())
        if missing:
            print(f"[Backbone] {len(missing)} keys not loaded (expected for v1b stem): "
                  f"{sorted(missing)[:5]} ...")
        model_sd.update(new_sd)
        model.load_state_dict(model_sd, strict=False)
        print("[Backbone] Pretrained weights loaded.")
    return model


if __name__ == "__main__":
    model = res2net50_v1b_26w_4s(pretrained=False)
    x = torch.randn(1, 3, 352, 352)
    outs = model(x)
    for i, o in enumerate(outs, 1):
        print(f"  layer{i}: {tuple(o.shape)}")
