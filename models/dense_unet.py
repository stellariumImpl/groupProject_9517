# models/dense_unet.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import densenet121


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return torch.cat([x, self.layer(x)], 1)


class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, n_layers):
        super(DenseBlock, self).__init__()
        self.blocks = nn.ModuleList([
            DenseLayer(in_channels + i * growth_rate, growth_rate)
            for i in range(n_layers)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x


class TransitionUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionUp, self).__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x, skip):
        out = self.conv(x)
        out = F.interpolate(out, size=skip.shape[2:], mode='bilinear', align_corners=True)
        return torch.cat([out, skip], 1)


class DenseUNet(nn.Module):
    def __init__(self, in_channels, num_classes, pretrained=True):
        super(DenseUNet, self).__init__()

        # Load pretrained DenseNet121
        densenet = densenet121(pretrained=pretrained)

        # Encoder (Downsampling)
        self.features = densenet.features

        # Adjust the first convolution layer if in_channels is not 3
        if in_channels != 3:
            self.features.conv0 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Decoder (Upsampling)
        self.decoder = nn.ModuleList()

        # Channel sizes for DenseNet121 (these might need adjustment based on actual output)
        encoder_channels = [256, 512, 1024, 1024]
        decoder_channels = [512, 256, 128]

        for i in range(3):
            trans = TransitionUp(encoder_channels[3 - i], decoder_channels[i])
            self.decoder.append(trans)
            in_channels = decoder_channels[i] + encoder_channels[2 - i]
            block = DenseBlock(in_channels, 32, 4)  # Reduced number of layers
            self.decoder.append(block)

        # Final convolution
        self.final_conv = nn.Conv2d(decoder_channels[-1] + 32 * 4, num_classes, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i in [4, 6, 8, 11]:  # Collect skip connections
                skip_connections.append(x)
                print(f"Encoder layer {i} output shape: {x.shape}")

        # Decoder
        for i, block in enumerate(self.decoder):
            if isinstance(block, TransitionUp):
                x = block(x, skip_connections.pop())
            else:
                x = block(x)
            print(f"Decoder block {i} output shape: {x.shape}")

        x = self.final_conv(x)
        print(f"Final output shape: {x.shape}")
        return x


# 在 train.py 中使用：
# model = DenseUNet(in_channels=3, num_classes=18, pretrained=True)

# 添加一个辅助函数来打印模型的通道数
# def print_model_channels(model):
#     def hook(module, input, output):
#         print(f"{module.__class__.__name__}: Input shape: {input[0].shape}, Output shape: {output.shape}")
#
#     for name, layer in model.named_modules():
#         if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, DenseBlock, TransitionUp)):
#             layer.register_forward_hook(hook)

# 在 train.py 中使用这个函数
# model = DenseUNet(in_channels=3, num_classes=18, pretrained=True).to(device)
# print_model_channels(model)
# dummy_input = torch.randn(1, 3, 256, 256).to(device)
# _ = model(dummy_input)