# utils/losses.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, inputs, targets):
        # inputs shape: [N, C, H, W]
        # targets shape: [N, H, W]

        # Ensure inputs are in the correct shape
        if len(inputs.shape) != 4:
            raise ValueError(f"Expected inputs to have 4 dimensions, but got {len(inputs.shape)}")

        num_classes = inputs.shape[1]

        # Apply softmax to inputs
        inputs = F.softmax(inputs, dim=1)

        # Flatten inputs and targets
        inputs = inputs.view(inputs.shape[0], inputs.shape[1], -1)  # [N, C, H*W]
        targets = targets.view(targets.shape[0], -1)  # [N, H*W]

        # Create one-hot encoding of targets
        targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()  # [N, H*W, C]
        targets_one_hot = targets_one_hot.permute(0, 2, 1)  # [N, C, H*W]

        # Compute intersection and union
        intersection = (inputs * targets_one_hot).sum(dim=[0, 2])
        union = inputs.sum(dim=[0, 2]) + targets_one_hot.sum(dim=[0, 2])

        # Compute Dice coefficient for each class
        dice = (2. * intersection + self.smooth) / (union + self.smooth)

        # Return mean Dice loss
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    def __init__(self, weight_focal=1.0, weight_dice=0.5):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss()
        self.dice_loss = DiceLoss()
        self.weight_focal = weight_focal
        self.weight_dice = weight_dice

    def forward(self, inputs, targets):
        focal = self.focal_loss(inputs, targets)
        dice = self.dice_loss(inputs, targets)
        return self.weight_focal * focal + self.weight_dice * dice


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss