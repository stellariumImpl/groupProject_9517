import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, norm_layer=nn.BatchNorm2d,activation_layer=nn.ReLU, dropout_rate=0.2):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(mid_channels),
            activation_layer(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            norm_layer(out_channels),
            activation_layer(inplace=True),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU, dropout_rate=0.2):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels, norm_layer=norm_layer, activation_layer=activation_layer, dropout_rate=dropout_rate)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU,
                 dropout_rate=0.2):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels, norm_layer=norm_layer, activation_layer=activation_layer,
                               dropout_rate=dropout_rate)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 确保输入 x1 和 x2 的形状相同
        x1 = F.interpolate(x1, size=x2.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()

        self.encoder1 = DoubleConv(in_channels, 64)
        self.encoder2 = Down(64, 128)
        self.encoder3 = Down(128 , 256)
        self.encoder4 = Down(256, 512)
        self.encoder5 = Down(512, 1024)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.upconv4 = Up(1024, 512, bilinear=True)
        self.upconv3 = Up(512, 256, bilinear=True)
        self.upconv2 = Up(256, 128, bilinear=True)
        self.upconv1 = Up(128, 64, bilinear=True)

        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)
        self.apply(init_weights)

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(enc1)
        enc3 = self.encoder3(enc2)
        enc4 = self.encoder4(enc3)
        enc5 = self.encoder5(enc4)

        dec4 = self.upconv4(enc5, enc4)
        dec3 = self.upconv3(dec4, enc3)
        dec2 = self.upconv2(dec3, enc2)
        dec1 = self.upconv1(dec2, enc1)

        return self.final_conv(dec1)
