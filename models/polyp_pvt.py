"""
Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformer.

Reference:
    Dong, B., Wang, W., Deng, D. P., Zhou, T., Shen, J., & Shao, L. (2023).
    Polyp-PVT: Polyp Segmentation with Pyramid Vision Transformer.
    CAAI Transactions on Intelligence Technology, 8(4), 989-1002.

Architecture:
    Encoder: PVTv2-B2 (Pyramid Vision Transformer v2)
    Modules:
        - CIM: Cascaded Attention Module (Channel + Spatial attention on shallow features)
        - CFM: Cascaded Feature Fusion Module (multi-scale feature aggregation)
        - SAM: Similarity-Aggregation Module (graph convolution reasoning)
    Outputs:
        - prediction1_8: Output from CFM (coarse map, upsampled to input size)
        - prediction2_8: Output from SAM (fine boundary map, upsampled to input size)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models import register_model


# ---------------------------------------------------------------------------
# Building Blocks
# ---------------------------------------------------------------------------

class BasicConv2d(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class CFM(nn.Module):
    """Cascaded Feature Fusion Module."""
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
        self.conv4 = BasicConv2d(3 * channel, channel, 3, padding=1)

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
        return self.conv4(x3_2)


class GCN(nn.Module):
    """Graph Convolutional Network block for spatial relationship reasoning."""
    def __init__(self, num_state: int, num_node: int, bias: bool = False):
        super().__init__()
        self.conv1 = nn.Conv1d(num_node, num_node, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(num_state, num_state, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x.permute(0, 2, 1)).permute(0, 2, 1)
        h = h - x
        return self.relu(self.conv2(h))


class SAM(nn.Module):
    """Similarity-Aggregation Module with GCN."""
    def __init__(self, num_in: int = 32, plane_mid: int = 16, mids: int = 4):
        super().__init__()
        self.num_s = int(plane_mid)
        self.num_n = mids * mids
        self.priors = nn.AdaptiveAvgPool2d(output_size=(mids + 2, mids + 2))

        self.conv_state = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.conv_proj = nn.Conv2d(num_in, self.num_s, kernel_size=1)
        self.gcn = GCN(num_state=self.num_s, num_node=self.num_n)
        self.conv_extend = nn.Conv2d(self.num_s, num_in, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        edge = F.interpolate(edge, (x.size(-2), x.size(-1)), mode='bilinear', align_corners=False)
        n = x.size(0)

        # Soft edge guidance (1 channel for spatial attention weighting)
        if edge.size(1) > 1:
            edge = torch.mean(edge, dim=1, keepdim=True)
        edge_weight = torch.sigmoid(edge)

        x_state_reshaped = self.conv_state(x).view(n, self.num_s, -1)
        x_proj = self.conv_proj(x)
        x_mask = x_proj * edge_weight

        x_anchor = self.priors(x_mask)[:, :, 1:-1, 1:-1].reshape(n, self.num_s, -1)

        x_proj_reshaped = torch.matmul(x_anchor.permute(0, 2, 1), x_proj.reshape(n, self.num_s, -1))
        x_proj_reshaped = torch.softmax(x_proj_reshaped, dim=1)

        x_n_state = torch.matmul(x_state_reshaped, x_proj_reshaped.permute(0, 2, 1))
        x_n_rel = self.gcn(x_n_state)

        x_state_reshaped = torch.matmul(x_n_rel, x_proj_reshaped)
        x_state = x_state_reshaped.view(n, self.num_s, *x.size()[2:])
        return x + self.conv_extend(x_state)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes: int, ratio: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        return self.sigmoid(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        return self.sigmoid(self.conv1(x_cat))


# ---------------------------------------------------------------------------
# Polyp-PVT Model
# ---------------------------------------------------------------------------

@register_model('polyp_pvt')
class PolypPVT(nn.Module):
    """Polyp-PVT: Pyramid Vision Transformer for Polyp Segmentation."""

    def __init__(self, channel: int = 32, pretrained: bool = True):
        super().__init__()
        self.backbone = timm.create_model('pvt_v2_b2', pretrained=pretrained, features_only=True)

        self.Translayer2_0 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel, 1)
        self.Translayer4_1 = BasicConv2d(512, channel, 1)

        self.CFM = CFM(channel)
        self.ca = ChannelAttention(64)
        self.sa = SpatialAttention()
        self.SAM = SAM(num_in=channel)

        self.down05 = nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=True)
        self.out_CFM = nn.Conv2d(channel, 1, 1)
        self.out_SAM = nn.Conv2d(channel, 1, 1)

    def forward(self, x: torch.Tensor):
        # Backbone yields 4 pyramid levels
        feats = self.backbone(x)
        x1, x2, x3, x4 = feats[0], feats[1], feats[2], feats[3]

        # Cascaded Attention Module (CIM)
        x1 = self.ca(x1) * x1
        cim_feature = self.sa(x1) * x1

        # Cascaded Feature Fusion Module (CFM)
        x2_t = self.Translayer2_1(x2)
        x3_t = self.Translayer3_1(x3)
        x4_t = self.Translayer4_1(x4)
        cfm_feature = self.CFM(x4_t, x3_t, x2_t)

        # Similarity-Aggregation Module (SAM)
        t2 = self.down05(self.Translayer2_0(cim_feature))
        sam_feature = self.SAM(cfm_feature, t2)

        prediction1 = self.out_CFM(cfm_feature)
        prediction2 = self.out_SAM(sam_feature)

        # Upsample both predictions to full input resolution (352x352)
        prediction1_8 = F.interpolate(prediction1, scale_factor=8, mode='bilinear', align_corners=False)
        prediction2_8 = F.interpolate(prediction2, scale_factor=8, mode='bilinear', align_corners=False)

        # Return multi-supervision outputs: coarse (CFM) and fine (SAM)
        return prediction1_8, prediction2_8
