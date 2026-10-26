"""
CMUNeXt: An Efficient Medical Image Segmentation Network based on Large Kernel and Skip Fusion.

Reference:
    Tang, F., Ding, J., Wang, L., Ning, C., & Zhou, S. K. (2024).
    CMUNeXt: An Efficient Medical Image Segmentation Network based on
    Large Kernel and Skip Fusion.
    arXiv:2308.01239
    https://github.com/FengheTan9/CMUNeXt

Architecture (faithfully from network/CMUNeXt.py):
    CMUNeXtBlock: depthwise large-kernel conv (k×k) + inverted bottleneck (×4) + residual
    Encoder: stem (conv_block) + 5 CMUNeXtBlock stages with MaxPool2d between them
    Decoder: 4 upsampling stages with Skip-Fusion blocks (grouped conv + inverted bottleneck)
    Output: 1×1 conv head

Variants (from source code factory functions):
    cmunext   (base): dims=[16,32,128,160,256], depths=[1,1,1,3,1], kernels=[3,3,7,7,7]
    cmunext_s (small): dims=[8,16,32,64,128],   depths=[1,1,1,1,1], kernels=[3,3,7,7,9]
    cmunext_l (large): dims=[32,64,128,256,512], depths=[1,1,1,6,3], kernels=[3,3,7,7,7]
"""

import torch
import torch.nn as nn
from models import register_model


# ---------------------------------------------------------------------------
# Building blocks (faithful to https://github.com/FengheTan9/CMUNeXt)
# ---------------------------------------------------------------------------

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    """
    CMUNeXt encoder block.
    Depthwise large-kernel conv (k×k) with residual, followed by
    inverted bottleneck (×4 expand) pointwise convolutions.
    Ends with a conv_block to change channel dimension.
    """

    def __init__(self, ch_in, ch_out, depth=1, k=3):
        super(CMUNeXtBlock, self).__init__()
        self.block = nn.Sequential(
            *[nn.Sequential(
                Residual(nn.Sequential(
                    # depthwise conv
                    nn.Conv2d(ch_in, ch_in, kernel_size=(k, k), groups=ch_in,
                              padding=(k // 2, k // 2)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in)
                )),
                nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in * 4),
                nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                nn.GELU(),
                nn.BatchNorm2d(ch_in)
            ) for i in range(depth)]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x):
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    """3×3 Conv → BN → ReLU."""

    def __init__(self, ch_in, ch_out):
        super(conv_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    """2× bilinear upsample + Conv3×3 + BN + ReLU."""

    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class fusion_conv(nn.Module):
    """
    Skip-Fusion block.
    Grouped conv (groups=2) to separately adapt encoder & decoder features,
    followed by inverted bottleneck (×4) pointwise convolutions with GELU + BN.
    """

    def __init__(self, ch_in, ch_out):
        super(fusion_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(ch_in, ch_in, kernel_size=3, stride=1, padding=1, groups=2, bias=True),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


# ---------------------------------------------------------------------------
# CMUNeXt network
# ---------------------------------------------------------------------------

class CMUNeXt(nn.Module):
    """
    CMUNeXt: U-shaped encoder-decoder with CMUNeXt blocks and Skip-Fusion.

    Args:
        input_channel: Number of input image channels (default: 3).
        num_classes: Number of output segmentation classes (default: 1).
        dims: Channel dimensions for each of the 5 encoder levels.
        depths: Number of CMUNeXt block repetitions at each level.
        kernels: Depthwise conv kernel size at each level.
    """

    def __init__(
        self,
        input_channel=3,
        num_classes=1,
        dims=None,
        depths=None,
        kernels=None,
    ):
        super(CMUNeXt, self).__init__()
        if dims is None:
            dims = [16, 32, 128, 160, 256]
        if depths is None:
            depths = [1, 1, 1, 3, 1]
        if kernels is None:
            kernels = [3, 3, 7, 7, 7]

        # Encoder
        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=input_channel, ch_out=dims[0])
        self.encoder1 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[0], depth=depths[0], k=kernels[0])
        self.encoder2 = CMUNeXtBlock(ch_in=dims[0], ch_out=dims[1], depth=depths[1], k=kernels[1])
        self.encoder3 = CMUNeXtBlock(ch_in=dims[1], ch_out=dims[2], depth=depths[2], k=kernels[2])
        self.encoder4 = CMUNeXtBlock(ch_in=dims[2], ch_out=dims[3], depth=depths[3], k=kernels[3])
        self.encoder5 = CMUNeXtBlock(ch_in=dims[3], ch_out=dims[4], depth=depths[4], k=kernels[4])

        # Decoder
        self.Up5 = up_conv(ch_in=dims[4], ch_out=dims[3])
        self.Up_conv5 = fusion_conv(ch_in=dims[3] * 2, ch_out=dims[3])
        self.Up4 = up_conv(ch_in=dims[3], ch_out=dims[2])
        self.Up_conv4 = fusion_conv(ch_in=dims[2] * 2, ch_out=dims[2])
        self.Up3 = up_conv(ch_in=dims[2], ch_out=dims[1])
        self.Up_conv3 = fusion_conv(ch_in=dims[1] * 2, ch_out=dims[1])
        self.Up2 = up_conv(ch_in=dims[1], ch_out=dims[0])
        self.Up_conv2 = fusion_conv(ch_in=dims[0] * 2, ch_out=dims[0])

        # Output head
        self.Conv_1x1 = nn.Conv2d(dims[0], num_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        # Encoder
        x1 = self.stem(x)
        x1 = self.encoder1(x1)
        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)
        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)
        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)
        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        # Decoder with Skip-Fusion
        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = self.Conv_1x1(d2)
        return d1


# ---------------------------------------------------------------------------
# Registered variants
# ---------------------------------------------------------------------------

@register_model('cmunext')
def _cmunext_base(img_ch: int = 3, output_ch: int = 1, **kwargs):
    """CMUNeXt base variant: dims=[16,32,128,160,256], depths=[1,1,1,3,1], kernels=[3,3,7,7,7]."""
    return CMUNeXt(
        input_channel=img_ch, num_classes=output_ch,
        dims=[16, 32, 128, 160, 256], depths=[1, 1, 1, 3, 1], kernels=[3, 3, 7, 7, 7],
    )


@register_model('cmunext_s')
def _cmunext_small(img_ch: int = 3, output_ch: int = 1, **kwargs):
    """CMUNeXt-S (small) variant: dims=[8,16,32,64,128], depths=[1,1,1,1,1], kernels=[3,3,7,7,9]."""
    return CMUNeXt(
        input_channel=img_ch, num_classes=output_ch,
        dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9],
    )


@register_model('cmunext_l')
def _cmunext_large(img_ch: int = 3, output_ch: int = 1, **kwargs):
    """CMUNeXt-L (large) variant: dims=[32,64,128,256,512], depths=[1,1,1,6,3], kernels=[3,3,7,7,7]."""
    return CMUNeXt(
        input_channel=img_ch, num_classes=output_ch,
        dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7],
    )


@register_model('cmunext_s')
def _cmunext_small(img_ch: int = 3, output_ch: int = 1, **kwargs):
    """CMUNeXt-S (small) variant: dims=[8,16,32,64,128], depths=[1,1,1,1,1], kernels=[3,3,7,7,9]."""
    return CMUNeXt(
        input_channel=img_ch, num_classes=output_ch,
        dims=[8, 16, 32, 64, 128], depths=[1, 1, 1, 1, 1], kernels=[3, 3, 7, 7, 9],
    )


@register_model('cmunext_l')
def _cmunext_large(img_ch: int = 3, output_ch: int = 1, **kwargs):
    """CMUNeXt-L (large) variant: dims=[32,64,128,256,512], depths=[1,1,1,6,3], kernels=[3,3,7,7,7]."""
    return CMUNeXt(
        input_channel=img_ch, num_classes=output_ch,
        dims=[32, 64, 128, 256, 512], depths=[1, 1, 1, 6, 3], kernels=[3, 3, 7, 7, 7],
    )

