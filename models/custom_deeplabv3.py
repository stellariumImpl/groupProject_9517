# models/custom_deeplabv3.py
import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet101
import torchvision.models as models

class CustomDeepLabV3(nn.Module):
    def __init__(self, num_classes):
        super(CustomDeepLabV3, self).__init__()
        # self.deeplabv3 = deeplabv3_resnet101(pretrained=True)
        self.deeplabv3 = deeplabv3_resnet101(weights=models.segmentation.DeepLabV3_ResNet101_Weights.DEFAULT)
        self.deeplabv3.classifier[4] = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Conv2d(256, num_classes, kernel_size=1)
        )

    def forward(self, x):
        return self.deeplabv3(x)