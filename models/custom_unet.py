import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(in_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))
        self.down4 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(512, 1024))
        self.up1 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv1 = DoubleConv(1024, 512)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv2 = DoubleConv(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv3 = DoubleConv(256, 128)
        self.up4 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv4 = DoubleConv(128, 64)
        self.outc = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        # print(f"Input shape: {x.shape}")
        x1 = self.inc(x)
        # print(f"After inc: {x1.shape}")
        x2 = self.down1(x1)
        # print(f"After down1: {x2.shape}")
        x3 = self.down2(x2)
        # print(f"After down2: {x3.shape}")
        x4 = self.down3(x3)
        # print(f"After down3: {x4.shape}")
        x5 = self.down4(x4)
        # print(f"After down4: {x5.shape}")
        x = self.up1(x5)
        # print(f"After up1: {x.shape}")
        x = torch.cat([x, x4], dim=1)
        x = self.conv1(x)
        # print(f"After conv1: {x.shape}")
        x = self.up2(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv2(x)
        # print(f"After conv2: {x.shape}")
        x = self.up3(x)
        x = torch.cat([x, x2], dim=1)
        x = self.conv3(x)
        # print(f"After conv3: {x.shape}")
        x = self.up4(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv4(x)
        # print(f"After conv4: {x.shape}")
        x = self.outc(x)
        # print(f"Final output shape: {x.shape}")
        return x
