"""CMUNeXt-Plus: experimental, checkpoint-compatible polyp-segmentation variant.

Inspired by (not a reproduction of): Attention U-Net (Oktay et al., 2018),
PraNet (Fan et al., 2020), Polyp-PVT (Dong et al., 2023), and CMUNeXt
(Tang et al., 2023/24). Performance improvement requires controlled tests.

The original CMUNeXt encoder, upsampling and fusion layers keep their state
names and shapes so a base `cmunext` checkpoint may initialise those layers.
Added modules start with zero residual gain, preserving the base output
before fine-tuning (in eval mode).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model
from models.cmunext import CMUNeXt


class MultiScaleContext(nn.Module):
    """Lightweight parallel dilated depthwise filters at the bottleneck."""

    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=d,
                          dilation=d, groups=channels, bias=False),
                nn.BatchNorm2d(channels),
                nn.GELU(),
            ) for d in (1, 2, 3)
        ])
        self.mix = nn.Sequential(
            nn.Conv2d(3 * channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )
        # Exact identity on day one for pretrained base weight compatibility.
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        context = self.mix(torch.cat([layer(x) for layer in self.branches], dim=1))
        return x + self.gain * context


class IdentityInitAttentionGate(nn.Module):
    """Decoder-conditioned spatial skip gate with identity initialization."""

    def __init__(self, channels: int):
        super().__init__()
        hidden = max(8, channels // 4)
        self.skip_project = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.gate_project = nn.Conv2d(channels, hidden, kernel_size=1, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.mask_project = nn.Conv2d(hidden, 1, kernel_size=1)
        self.gain = nn.Parameter(torch.zeros(()))

    def forward(self, skip: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
        if up.shape[-2:] != skip.shape[-2:]:
            up = F.interpolate(up, size=skip.shape[-2:], mode='bilinear',
                               align_corners=False)
        logits = self.mask_project(self.act(self.skip_project(skip) + self.gate_project(up)))
        # gain==0 gives skip unchanged, unlike typical sigmoid attention gates.
        weight = 2.0 * torch.sigmoid(logits) - 1.0
        return skip * (1.0 + self.gain * weight)


class CMUNeXtPlus(CMUNeXt):
    """CMUNeXt with context pyramid, gated skips and deep supervision.

    At training time returns (aux_coarse, aux_medium, aux_fine, final),
    with final full-resolution prediction LAST to match Trainer/Evaluator.
    """

    def __init__(self):
        dims = [16, 32, 128, 160, 256]
        super().__init__(input_channel=3, num_classes=1,
                         dims=dims, depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7])
        self.context = MultiScaleContext(dims[4])
        self.gate5 = IdentityInitAttentionGate(dims[3])
        self.gate4 = IdentityInitAttentionGate(dims[2])
        self.gate3 = IdentityInitAttentionGate(dims[1])
        self.gate2 = IdentityInitAttentionGate(dims[0])
        self.aux5 = nn.Conv2d(dims[3], 1, kernel_size=1)
        self.aux4 = nn.Conv2d(dims[2], 1, kernel_size=1)
        self.aux3 = nn.Conv2d(dims[1], 1, kernel_size=1)

    def forward(self, x: torch.Tensor):
        full_size = x.shape[-2:]
        x1 = self.encoder1(self.stem(x))
        x2 = self.encoder2(self.Maxpool(x1))
        x3 = self.encoder3(self.Maxpool(x2))
        x4 = self.encoder4(self.Maxpool(x3))
        x5 = self.context(self.encoder5(self.Maxpool(x4)))

        u5 = self.Up5(x5)
        d5 = self.Up_conv5(torch.cat((self.gate5(x4, u5), u5), dim=1))
        u4 = self.Up4(d5)
        d4 = self.Up_conv4(torch.cat((self.gate4(x3, u4), u4), dim=1))
        u3 = self.Up3(d4)
        d3 = self.Up_conv3(torch.cat((self.gate3(x2, u3), u3), dim=1))
        u2 = self.Up2(d3)
        d2 = self.Up_conv2(torch.cat((self.gate2(x1, u2), u2), dim=1))
        main = self.Conv_1x1(d2)
        # Match input size for scalar segmentation losses across all levels.
        return (
            F.interpolate(self.aux5(d5), size=full_size, mode='bilinear', align_corners=False),
            F.interpolate(self.aux4(d4), size=full_size, mode='bilinear', align_corners=False),
            F.interpolate(self.aux3(d3), size=full_size, mode='bilinear', align_corners=False),
            main,
        )


@register_model('cmunext_plus')
def make_cmunext_plus():
    return CMUNeXtPlus()