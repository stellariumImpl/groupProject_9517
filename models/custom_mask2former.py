import torch
import torch.nn as nn
from transformers import Mask2FormerForUniversalSegmentation

class CustomMask2Former(nn.Module):
    def __init__(self, num_classes):
        super(CustomMask2Former, self).__init__()
        self.mask2former = Mask2FormerForUniversalSegmentation.from_pretrained(
            "facebook/mask2former-swin-base-coco-panoptic")
        
        # freezing most part of parameters
        for param in self.mask2former.parameters():
            param.requires_grad = False
        
        # unfrozen the last conv layer
        trainable_layers = list(self.mask2former.named_children())[-5:]
        for name, layer in trainable_layers:
            for param in layer.parameters():
                param.requires_grad = True

        # modify the output layer to accommodate the number of classes
        self.mask2former.config.num_labels = num_classes
        if hasattr(self.mask2former, 'class_predictor'):
            in_features = self.mask2former.class_predictor.in_features
            self.mask2former.class_predictor = nn.Linear(in_features, num_classes)

        # adding additional convolution layers
        self.extra_conv = nn.Conv2d(100, num_classes, kernel_size=3, padding=1)

    def forward(self, pixel_values):
        outputs = self.mask2former(pixel_values=pixel_values)
        masks = outputs.masks_queries_logits  # Shape: [batch_size, 100, height, width]
        masks = self.extra_conv(masks)  # Shape: [batch_size, num_classes, height, width]
        return masks
