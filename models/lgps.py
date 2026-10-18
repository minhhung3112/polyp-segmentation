import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2
import torch.nn.functional as F
from models import register_model

class SqueezeExcitation(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ModifiedResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ModifiedResidualBlock, self).__init__()
        
        mid_channels = out_channels // 4
        
        self.conv1 = nn.Conv2d(in_channels, mid_channels, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        
        if in_channels != mid_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels)
            )
        else:
            self.shortcut = nn.Identity()
            
        self.se = SqueezeExcitation(mid_channels)
        
    def forward(self, x):
        residual = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        out = self.se(out)
        return out

@register_model('lgps')
class LGPSGenerator(nn.Module):
    def __init__(self, pretrained=True):
        super(LGPSGenerator, self).__init__()
        
        backbone = mobilenet_v2(pretrained=pretrained)
        self.features = backbone.features
        
        self.bottleneck = ModifiedResidualBlock(576, 576)
        
        self.up1_block = ModifiedResidualBlock(144 + 192, 256) 
        
        self.up2_block = ModifiedResidualBlock(64 + 144, 128) 
        
        self.up3_block = ModifiedResidualBlock(32 + 96, 64) 
        
        self.out_conv = nn.Conv2d(16, 1, 1)
        
    def _get_expand_relu(self, x, block):
        out = block.conv[0](x)
        return out
        
    def forward(self, x):
        features = self.features
        
        out = features[0](x)
        out = features[1](out)
        
        skip1 = self._get_expand_relu(out, features[2])
        out = features[2](out)
        out = features[3](out)
        
        skip2 = self._get_expand_relu(out, features[4])
        out = features[4](out)
        out = features[5](out)
        out = features[6](out)
        
        skip3 = self._get_expand_relu(out, features[7])
        out = features[7](out)
        out = features[8](out)
        out = features[9](out)
        out = features[10](out)
        out = features[11](out)
        out = features[12](out)
        out = features[13](out)
        
        enc_out = self._get_expand_relu(out, features[14])
        
        bot = self.bottleneck(enc_out)
        
        up1 = F.interpolate(bot, scale_factor=2, mode='bilinear', align_corners=False)
        up1 = torch.cat([up1, skip3], dim=1)
        up1 = self.up1_block(up1)
        
        up2 = F.interpolate(up1, scale_factor=2, mode='bilinear', align_corners=False)
        up2 = torch.cat([up2, skip2], dim=1)
        up2 = self.up2_block(up2)
        
        up3 = F.interpolate(up2, scale_factor=2, mode='bilinear', align_corners=False)
        up3 = torch.cat([up3, skip1], dim=1)
        up3 = self.up3_block(up3)
        
        up4 = F.interpolate(up3, scale_factor=2, mode='bilinear', align_corners=False)
        out = self.out_conv(up4)
        
        return out
