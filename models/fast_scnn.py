"""
Fast-SCNN: Fast Segmentation Convolutional Neural Network.

Reference:
    Poudel, R. P., Liwicki, M., & Cipolla, R. (2019).
    Fast-SCNN: Fast Segmentation Convolutional Neural Network.
    British Machine Vision Conference (BMVC 2019).
    https://arxiv.org/abs/1902.04502
    https://github.com/Tramac/Fast-SCNN-pytorch

Architecture:
    1. Learning to Downsample (LDS): Conv-BN-ReLU + 2 Depthwise Separable Convolutions
    2. Global Feature Extractor (GFE): Inverted Residual Bottlenecks + Pyramid Pooling Module (PPM)
    3. Feature Fusion Module (FFM): Multi-resolution feature combination
    4. Classifier: 2 Depthwise Separable Convolutions + 1x1 Conv Head
    Params: ~1.14M (1.1357M parameters, matches paper ~1.11M-1.14M).

Registered as: 'fast_scnn', 'fastscnn'
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import register_model


# ---------------------------------------------------------------------------
# Fast-SCNN Building Blocks (from https://github.com/Tramac/Fast-SCNN-pytorch)
# ---------------------------------------------------------------------------

class _ConvBNReLU(nn.Module):
    """Conv2d -> BatchNorm2d -> ReLU."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _DSConv(nn.Module):
    """Depthwise Separable Convolution."""

    def __init__(self, dw_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class _DWConv(nn.Module):
    """Depthwise Convolution."""

    def __init__(self, dw_channels, out_channels, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """Linear Bottleneck (Inverted Residual Block)."""

    def __init__(self, in_channels, out_channels, t=6, stride=2):
        super().__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            _ConvBNReLU(in_channels, in_channels * t, 1),
            _DWConv(in_channels * t, in_channels * t, stride),
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out


class PyramidPooling(nn.Module):
    """Pyramid Pooling Module (PPM)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = _ConvBNReLU(in_channels, inter_channels, 1)
        self.conv2 = _ConvBNReLU(in_channels, inter_channels, 1)
        self.conv3 = _ConvBNReLU(in_channels, inter_channels, 1)
        self.conv4 = _ConvBNReLU(in_channels, inter_channels, 1)
        self.out = _ConvBNReLU(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        return nn.AdaptiveAvgPool2d(size)(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        return self.out(x)


class LearningToDownsample(nn.Module):
    """Learning to Downsample Module (LDS)."""

    def __init__(self, dw_channels1=32, dw_channels2=48, out_channels=64):
        super().__init__()
        self.conv = _ConvBNReLU(3, dw_channels1, 3, 2, padding=1)
        self.dsconv1 = _DSConv(dw_channels1, dw_channels2, 2)
        self.dsconv2 = _DSConv(dw_channels2, out_channels, 2)

    def forward(self, x):
        return self.dsconv2(self.dsconv1(self.conv(x)))


class GlobalFeatureExtractor(nn.Module):
    """Global Feature Extractor Module (GFE)."""

    def __init__(self, in_channels=64, block_channels=(64, 96, 128),
                 out_channels=128, t=6, num_blocks=(3, 3, 3)):
        super().__init__()
        self.bottleneck1 = self._make_layer(LinearBottleneck, in_channels, block_channels[0], num_blocks[0], t, 2)
        self.bottleneck2 = self._make_layer(LinearBottleneck, block_channels[0], block_channels[1], num_blocks[1], t, 2)
        self.bottleneck3 = self._make_layer(LinearBottleneck, block_channels[1], block_channels[2], num_blocks[2], t, 1)
        self.ppm = PyramidPooling(block_channels[2], out_channels)

    def _make_layer(self, block, inplanes, planes, blocks, t=6, stride=1):
        layers = [block(inplanes, planes, t, stride)]
        for _ in range(1, blocks):
            layers.append(block(planes, planes, t, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.bottleneck1(x)
        x = self.bottleneck2(x)
        x = self.bottleneck3(x)
        return self.ppm(x)


class FeatureFusionModule(nn.Module):
    """Feature Fusion Module (FFM)."""

    def __init__(self, higher_in_channels, lower_in_channels, out_channels):
        super().__init__()
        self.dwconv = _DWConv(lower_in_channels, out_channels, 1)
        self.conv_lower_res = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.conv_higher_res = nn.Sequential(
            nn.Conv2d(higher_in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, higher_res_feature, lower_res_feature):
        target_size = higher_res_feature.shape[2:]
        lower_res_feature = F.interpolate(lower_res_feature, size=target_size, mode='bilinear', align_corners=True)
        lower_res_feature = self.conv_lower_res(self.dwconv(lower_res_feature))
        higher_res_feature = self.conv_higher_res(higher_res_feature)
        return self.relu(higher_res_feature + lower_res_feature)


class Classifier(nn.Module):
    """Classifier."""

    def __init__(self, dw_channels, num_classes, stride=1):
        super().__init__()
        self.dsconv1 = _DSConv(dw_channels, dw_channels, stride)
        self.dsconv2 = _DSConv(dw_channels, dw_channels, stride)
        self.conv = nn.Sequential(
            nn.Dropout(0.1),
            nn.Conv2d(dw_channels, num_classes, 1),
        )

    def forward(self, x):
        x = self.dsconv1(x)
        x = self.dsconv2(x)
        return self.conv(x)


# ---------------------------------------------------------------------------
# Fast-SCNN Model
# ---------------------------------------------------------------------------

@register_model('fast_scnn')
@register_model('fastscnn')
class FastSCNN(nn.Module):
    """
    Fast-SCNN (BMVC 2019): Fast Segmentation Convolutional Neural Network.

    Parameters: ~1.14M (1.1357M params, paper target ~1.11M).
    Returns raw logits of shape (B, num_classes, H, W).
    """

    def __init__(self, num_classes: int = 1, aux: bool = False):
        super().__init__()
        self.aux = aux
        self.learning_to_downsample = LearningToDownsample(32, 48, 64)
        self.global_feature_extractor = GlobalFeatureExtractor(64, [64, 96, 128], 128, 6, [3, 3, 3])
        self.feature_fusion = FeatureFusionModule(64, 128, 128)
        self.classifier = Classifier(128, num_classes)

        if self.aux:
            self.auxlayer = nn.Sequential(
                nn.Conv2d(64, 32, 3, padding=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Conv2d(32, num_classes, 1),
            )

    def forward(self, x):
        orig_size = x.shape[2:]
        higher_res = self.learning_to_downsample(x)
        gfe_feat = self.global_feature_extractor(higher_res)
        fused = self.feature_fusion(higher_res, gfe_feat)
        out = self.classifier(fused)
        out = F.interpolate(out, size=orig_size, mode='bilinear', align_corners=True)

        if self.aux:
            aux_out = self.auxlayer(higher_res)
            aux_out = F.interpolate(aux_out, size=orig_size, mode='bilinear', align_corners=True)
            return aux_out, out  # finest prediction last

        return out
