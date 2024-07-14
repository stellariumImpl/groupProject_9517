# models/custom_mask2former.py
import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation


class CustomMask2Former(nn.Module):
    def __init__(self, num_classes):
        super(CustomMask2Former, self).__init__()

        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-coco-panoptic")

        self.mask2former.config.num_labels = num_classes

        # 修改输出层以适应类别数
        if hasattr(self.mask2former, 'class_predictor'):
            in_features = self.mask2former.class_predictor.in_features
            self.mask2former.class_predictor = nn.Linear(in_features, num_classes)

    def forward(self, pixel_values):
        return self.mask2former(pixel_values=pixel_values)