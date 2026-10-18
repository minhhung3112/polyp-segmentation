"""
PraNet — Parallel Reverse Attention Network for Polyp Segmentation.

Reference:
    Fan, D. P., Ji, G. P., Zhou, T., Chen, G., Fu, H., Shen, J., & Shao, L. (2020).
    PraNet: Parallel Reverse Attention Network for Polyp Segmentation.
    MICCAI 2020, LNCS 12266, pp. 263-273.

Architecture:
    Encoder : Res2Net-50 v1b (ImageNet pretrained)
    RFB     : Receptive Field Block (modified, 4 dilated branches)
    PD      : Partial Decoder — dense aggregation of x4, x3, x2 features
    RA      : Reverse Attention branches (x4 → x3 → x2, iterative refinement)
    Outputs : lateral_map_5 (coarse), lateral_map_4, lateral_map_3, lateral_map_2 (fine)

Usage:
    from models import get_model
    model = get_model('pranet')   # returns PraNet()
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models._backbone.res2net import res2net50_v1b_26w_4s
from models import register_model


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    """Conv2d → BN (no activation, caller decides when to apply ReLU)."""

    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class RFB_modified(nn.Module):
    """Receptive Field Block (modified) with 4 dilated parallel branches.

    Branches use strip convolutions + dilated 3×3 to enlarge receptive fields
    at dilation rates 1, 3, 5, 7 respectively.

    Args:
        in_channel  : Input channel count.
        out_channel : Output channel count per branch (total = 4 × out_channel).
    """

    def __init__(self, in_channel: int, out_channel: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(True)

        # Branch 0: plain 1×1
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )

        # Branch 1: 1×3 → 3×1 → dilated 3 (rate=3)
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3),
        )

        # Branch 2: 1×5 → 5×1 → dilated 5 (rate=5)
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5),
        )

        # Branch 3: 1×7 → 7×1 → dilated 7 (rate=7)
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7),
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
    """Dense Partial Decoder — aggregates x4, x3, x2 features.

    Implements the cascade partial decoder (CPD) design:
        - x4 upsampled ×2 → element-wise product with x3 (guidance)
        - combined upsampled ×2 → element-wise product with x2
        - concat → conv → single-channel output

    Args:
        channel: Unified channel width for all branches.
    """

    def __init__(self, channel: int) -> None:
        super().__init__()
        self.relu = nn.ReLU(True)
        self.upsample = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2 * channel, 2 * channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv4 = BasicConv2d(3 * channel, 3 * channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3 * channel, 1, 1)

    def forward(
        self,
        x1: torch.Tensor,  # deepest (e.g., x4_rfb)
        x2: torch.Tensor,  # middle  (e.g., x3_rfb)
        x3: torch.Tensor,  # shallow (e.g., x2_rfb)
    ) -> torch.Tensor:
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2
        x3_1 = (
            self.conv_upsample2(self.upsample(self.upsample(x1)))
            * self.conv_upsample3(self.upsample(x2))
            * x3
        )

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), dim=1)
        x2_2 = self.conv_concat2(x2_2)

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), dim=1)
        x3_2 = self.conv_concat3(x3_2)

        x = self.conv4(x3_2)
        return self.conv5(x)


# ---------------------------------------------------------------------------
# PraNet
# ---------------------------------------------------------------------------

@register_model("pranet")
class PraNet(nn.Module):
    """Parallel Reverse Attention Network for Polyp Segmentation.

    Forward returns four lateral maps at 352×352 resolution:
        lateral_map_5 : coarse (from Partial Decoder)
        lateral_map_4 : reverse attention branch 4 (deepest)
        lateral_map_3 : reverse attention branch 3
        lateral_map_2 : fine   (reverse attention branch 2)

    Args:
        channel: Unified channel width after RFB projection (default: 32).
    """

    def __init__(self, channel: int = 32) -> None:
        super().__init__()

        # ---- Res2Net-50 v1b Backbone ----
        self.resnet = res2net50_v1b_26w_4s(pretrained=True)
        # Backbone output channels: [256, 512, 1024, 2048]

        # ---- Receptive Field Blocks ----
        self.rfb2_1 = RFB_modified(512, channel)
        self.rfb3_1 = RFB_modified(1024, channel)
        self.rfb4_1 = RFB_modified(2048, channel)

        # ---- Partial Decoder (coarse prediction) ----
        self.agg1 = aggregation(channel)

        # ---- Reverse Attention Branch 4 (x4: 2048 ch) ----
        self.ra4_conv1 = BasicConv2d(2048, 256, kernel_size=1)
        self.ra4_conv2 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv3 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv4 = BasicConv2d(256, 256, kernel_size=5, padding=2)
        self.ra4_conv5 = BasicConv2d(256, 1,   kernel_size=1)

        # ---- Reverse Attention Branch 3 (x3: 1024 ch) ----
        self.ra3_conv1 = BasicConv2d(1024, 64, kernel_size=1)
        self.ra3_conv2 = BasicConv2d(64,  64, kernel_size=3, padding=1)
        self.ra3_conv3 = BasicConv2d(64,  64, kernel_size=3, padding=1)
        self.ra3_conv4 = BasicConv2d(64,   1, kernel_size=3, padding=1)

        # ---- Reverse Attention Branch 2 (x2: 512 ch) ----
        self.ra2_conv1 = BasicConv2d(512, 64, kernel_size=1)
        self.ra2_conv2 = BasicConv2d(64,  64, kernel_size=3, padding=1)
        self.ra2_conv3 = BasicConv2d(64,  64, kernel_size=3, padding=1)
        self.ra2_conv4 = BasicConv2d(64,   1, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Forward pass.

        Args:
            x: Input tensor ``(B, 3, H, W)``, typically H=W=352.

        Returns:
            Tuple of four prediction maps, all at the input resolution:
            ``(lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2)``.
        """
        pvt = self.resnet(x)
        x1 = pvt[0]   # (B, 256,  88,  88)
        x2 = pvt[1]   # (B, 512,  44,  44)
        x3 = pvt[2]   # (B, 1024, 22,  22)
        x4 = pvt[3]   # (B, 2048, 11,  11)

        # --- RFB projection ---
        x2_rfb = self.rfb2_1(x2)   # (B, 32, 44, 44)
        x3_rfb = self.rfb3_1(x3)   # (B, 32, 22, 22)
        x4_rfb = self.rfb4_1(x4)   # (B, 32, 11, 11)

        # --- Partial Decoder (coarse map Sup-1) ---
        ra5_feat = self.agg1(x4_rfb, x3_rfb, x2_rfb)          # (B, 1, 44, 44)
        lateral_map_5 = F.interpolate(
            ra5_feat, scale_factor=8, mode="bilinear", align_corners=False
        )  # (B, 1, 352, 352)

        # --- Reverse Attention Branch 4 ---
        crop_4 = F.interpolate(ra5_feat, scale_factor=0.25, mode="bilinear", align_corners=False)
        x = (-1 * torch.sigmoid(crop_4) + 1).expand(-1, 2048, -1, -1) * x4
        x = self.ra4_conv1(x)
        x = F.relu(self.ra4_conv2(x), inplace=True)
        x = F.relu(self.ra4_conv3(x), inplace=True)
        x = F.relu(self.ra4_conv4(x), inplace=True)
        ra4_feat = self.ra4_conv5(x)
        x = ra4_feat + crop_4
        lateral_map_4 = F.interpolate(
            x, scale_factor=32, mode="bilinear", align_corners=False
        )  # (B, 1, 352, 352)

        # --- Reverse Attention Branch 3 ---
        crop_3 = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = (-1 * torch.sigmoid(crop_3) + 1).expand(-1, 1024, -1, -1) * x3
        x = self.ra3_conv1(x)
        x = F.relu(self.ra3_conv2(x), inplace=True)
        x = F.relu(self.ra3_conv3(x), inplace=True)
        ra3_feat = self.ra3_conv4(x)
        x = ra3_feat + crop_3
        lateral_map_3 = F.interpolate(
            x, scale_factor=16, mode="bilinear", align_corners=False
        )  # (B, 1, 352, 352)

        # --- Reverse Attention Branch 2 ---
        crop_2 = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = (-1 * torch.sigmoid(crop_2) + 1).expand(-1, 512, -1, -1) * x2
        x = self.ra2_conv1(x)
        x = F.relu(self.ra2_conv2(x), inplace=True)
        x = F.relu(self.ra2_conv3(x), inplace=True)
        ra2_feat = self.ra2_conv4(x)
        x = ra2_feat + crop_2
        lateral_map_2 = F.interpolate(
            x, scale_factor=8, mode="bilinear", align_corners=False
        )  # (B, 1, 352, 352)

        return lateral_map_5, lateral_map_4, lateral_map_3, lateral_map_2


if __name__ == "__main__":
    model = PraNet().cuda()
    x = torch.randn(1, 3, 352, 352).cuda()
    outs = model(x)
    for i, o in enumerate(outs, 5):
        print(f"  lateral_map_{i}: {tuple(o.shape)}")
