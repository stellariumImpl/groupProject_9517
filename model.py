import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50, deeplabv3_resnet101, deeplabv3_mobilenet_v3_large
from torchvision.models.segmentation.deeplabv3 import DeepLabHead

class CustomDeepLabHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(CustomDeepLabHead, self).__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        return self.head(x)

def create_model(num_classes, backbone='resnet50', custom_head=False):
    """
    创建一个基于 DeepLabV3 的模型，可选择不同的骨干网络和自定义头部。

    参数:
        num_classes (int): 分割任务的类别数量
        backbone (str): 选择骨干网络，可选 'resnet50', 'resnet101', 'mobilenet_v3_large'
        custom_head (bool): 是否使用自定义头部网络

    返回:
        model (nn.Module): 配置好的 DeepLabV3 模型
    """
    if backbone == 'resnet50':
        model = deeplabv3_resnet50(pretrained=True)
    elif backbone == 'resnet101':
        model = deeplabv3_resnet101(pretrained=True)
    elif backbone == 'mobilenet_v3_large':
        model = deeplabv3_mobilenet_v3_large(pretrained=True)
    else:
        raise ValueError("Unsupported backbone. Choose 'resnet50', 'resnet101', or 'mobilenet_v3_large'")

    in_channels = 2048 if backbone in ['resnet50', 'resnet101'] else 960  # mobilenet_v3_large 使用 960 通道

    if custom_head:
        model.classifier = CustomDeepLabHead(in_channels, num_classes)
    else:
        model.classifier = DeepLabHead(in_channels, num_classes)

    return model

def train_model(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)

            outputs = model(images)['out']
            loss = criterion(outputs, masks)

            total_loss += loss.item()

    return total_loss / len(val_loader)

def train_and_validate(model, train_loader, val_loader, num_epochs, criterion, optimizer, device):
    for epoch in range(num_epochs):
        train_loss = train_model(model, train_loader, criterion, optimizer, device)
        val_loss = validate_model(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}:")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")
        print("-----------------------------")

    print("Training completed!")