import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision.transforms.functional import rgb_to_grayscale
from models._backbone.res2net import res2net50_v1b_26w_4s
from models import register_model

def gauss_kernel(channels=3, cuda=True):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                            [4., 16., 24., 16., 4.],
                            [6., 24., 36., 24., 6.],
                            [4., 16., 24., 16., 4.],
                            [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    if cuda:
        kernel = kernel.cuda()
    return kernel

def downsample(x):
    return x[:, :, ::2, ::2]

def conv_gauss(img, kernel):
    img = F.pad(img, (2, 2, 2, 2), mode='reflect')
    # kernel needs to be on the same device as img
    kernel = kernel.to(img.device)
    out = F.conv2d(img, kernel, groups=img.shape[1])
    return out

def upsample(x, channels):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2] * 2, x.shape[3])
    cc = cc.permute(0, 1, 3, 2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2] * 2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3] * 2, x.shape[2] * 2)
    x_up = cc.permute(0, 1, 3, 2)
    kernel = 4 * gauss_kernel(channels, cuda=False).to(x.device)
    return conv_gauss(x_up, kernel)

def make_laplace(img, channels):
    kernel = gauss_kernel(channels, cuda=False).to(img.device)
    filtered = conv_gauss(img, kernel)
    down = downsample(filtered)
    up = upsample(down, channels)
    if up.shape[2] != img.shape[2] or up.shape[3] != img.shape[3]:
        up = nn.functional.interpolate(up, size=(img.shape[2], img.shape[3]), mode='bilinear', align_corners=False)
    diff = img - up
    return diff

def make_laplace_pyramid(img, level, channels):
    current = img
    pyr = []
    for _ in range(level):
        kernel = gauss_kernel(channels, cuda=False).to(img.device)
        filtered = conv_gauss(current, kernel)
        down = downsample(filtered)
        up = upsample(down, channels)
        if up.shape[2] != current.shape[2] or up.shape[3] != current.shape[3]:
            up = nn.functional.interpolate(up, size=(current.shape[2], current.shape[3]), mode='bilinear', align_corners=False)
        diff = current - up
        pyr.append(diff)
        current = down
    pyr.append(current)
    return pyr

class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
    def forward(self, x):
        avg_out = self.mlp(F.avg_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        max_out = self.mlp(F.max_pool2d( x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3))))
        channel_att_sum = avg_out + max_out
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.spatial = nn.Conv2d(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2)
    def forward(self, x):
        x_compress = torch.cat((torch.max(x,1)[0].unsqueeze(1), torch.mean(x,1).unsqueeze(1)), dim=1)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)
        return x * scale

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio)
        self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        x_out = self.SpatialGate(x_out)
        return x_out

class EGA(nn.Module):
    def __init__(self, in_channels):
        super(EGA, self).__init__()
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 3 , 1, 1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True))
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, 1, 3, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid())
        self.cbam = CBAM(in_channels)
    def forward(self, edge_feature, x, pred):
        residual = x
        xsize = x.size()[2:]
        pred = torch.sigmoid(pred)
        background_att = 1 - pred
        background_x= x * background_att
        edge_pred = make_laplace(pred, 1)  
        pred_feature = x * edge_pred
        edge_input = F.interpolate(edge_feature, size=xsize, mode='bilinear', align_corners=True)
        input_feature = x * edge_input
        fusion_feature = torch.cat([background_x, pred_feature, input_feature], dim=1)
        fusion_feature = self.fusion_conv(fusion_feature)
        attention_map = self.attention(fusion_feature)
        fusion_feature = fusion_feature * attention_map
        out = fusion_feature + residual
        out = self.cbam(out)
        return out

class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            Conv(in_channels, in_channels // 4), 
            Conv(in_channels // 4, out_channels)
        )
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        out = self.conv(x)
        return self.up(out)

class Out(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Out, self).__init__()
        self.conv1 = Conv(in_channels, in_channels // 4, kernel_size=kernel_size,
                               stride=stride, padding=padding)
        self.conv2 = nn.Conv2d(in_channels // 4, out_channels, 1)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class EGANet(nn.Module):
    def __init__(self, n_channels, n_classes, pretrained=True):
        super(EGANet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        
        resnet = res2net50_v1b_26w_4s(pretrained=pretrained)
        self.encoder1_conv = resnet.conv1
        self.encoder1_bn = resnet.bn1
        self.encoder1_relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.encoder2 = resnet.layer1
        self.encoder3 = resnet.layer2
        self.encoder4 = resnet.layer3
        self.encoder5 = resnet.layer4
        
        self.x5_dem_1 = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1), nn.BatchNorm2d(512), nn.ReLU(inplace=True))
        self.x4_dem_1 = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(inplace=True))
        self.x3_dem_1 = nn.Sequential(nn.Conv2d(512, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True))
        self.x2_dem_1 = nn.Sequential(nn.Conv2d(256, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))
        
        self.up5 = nn.Sequential(
            Conv(512, 512),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        ) 
        self.up4 = Up(768, 256)
        self.up3 = Up(384, 128)
        self.up2 = Up(192, 64)
        self.up1 = Up(128, 64)

        self.ega1 = EGA(64)
        self.ega2 = EGA(64)
        self.ega3 = EGA(128)
        self.ega4 = EGA(256)
        
        self.out5 = Out(512, n_classes)
        self.out4 = Out(256, n_classes)
        self.out3 = Out(128, n_classes)
        self.out2 = Out(64, n_classes)
        self.out1 = Out(64, n_classes)

    def forward(self, x):
        if x.shape[1] == 3:
            grayscale_img = rgb_to_grayscale(x)
        else:
            grayscale_img = x
        edge_feature = make_laplace_pyramid(grayscale_img, 5, 1)
        edge_feature = edge_feature[1]

        e1 = self.encoder1_conv(x)  
        e1 = self.encoder1_bn(e1)
        e1 = self.encoder1_relu(e1)
        e1_pool = self.maxpool(e1)  
        
        e2 = self.encoder2(e1_pool)
        e3 = self.encoder3(e2)  
        e4 = self.encoder4(e3)  
        e5 = self.encoder5(e4)  
        
        e5_dem_1 = self.x5_dem_1(e5)
        e4_dem_1 = self.x4_dem_1(e4)
        e3_dem_1 = self.x3_dem_1(e3)
        e2_dem_1 = self.x2_dem_1(e2)

        d5 = self.up5(e5_dem_1)
        out5 = self.out5(d5)
        ega4 = self.ega4(edge_feature, e4_dem_1, out5)

        d4 = self.up4(d5, ega4)
        out4 = self.out4(d4)    
        ega3 = self.ega3(edge_feature, e3_dem_1, out4)

        d3 = self.up3(d4, ega3)
        out3 = self.out3(d3)
        ega2 = self.ega2(edge_feature, e2_dem_1, out3)

        d2 = self.up2(d3, ega2)
        out2 = self.out2(d2)  
        ega1 = self.ega1(edge_feature, e1, out2)

        d1 = self.up1(d2, ega1)
        out1 = self.out1(d1)
        
        return out5, out4, out3, out2, out1 # reversed to return finest last

@register_model('meganet')
class EGANetModel(nn.Module): 
    def __init__(self, pretrained=True):
        super(EGANetModel,self).__init__()
        self.net = EGANet(3, 1, pretrained=pretrained)

    def forward(self, images):
        return self.net(images)
