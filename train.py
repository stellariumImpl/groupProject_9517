import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from data_load import EnhancedWildScenesDataset
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import confusion_matrix


def calculate_miou(pred, target, num_classes):
    pred = pred.flatten()
    target = target.flatten()
    cm = confusion_matrix(target, pred, labels=range(num_classes))
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)
    iou = intersection / union.astype(np.float32)
    return np.mean(iou)


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    total_miou = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        pred = torch.argmax(outputs, dim=1)
        miou = calculate_miou(pred.cpu().numpy(), labels.cpu().numpy(), num_classes=18)
        total_miou += miou

    return total_loss / len(dataloader), total_miou / len(dataloader)


def validate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    total_miou = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)['out']

            loss = criterion(outputs, labels)
            total_loss += loss.item()

            pred = torch.argmax(outputs, dim=1)
            miou = calculate_miou(pred.cpu().numpy(), labels.cpu().numpy(), num_classes=18)
            total_miou += miou

    return total_loss / len(dataloader), total_miou / len(dataloader)


def save_checkpoint(model, optimizer, epoch, miou, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'miou': miou,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved: {filename}")


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir):
    best_miou = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_miou = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Training Loss: {train_loss:.4f}, Training mIoU: {train_miou:.4f}")

        val_loss, val_miou = validate_epoch(model, val_loader, criterion, device)
        print(f"Validation Loss: {val_loss:.4f}, Validation mIoU: {val_miou:.4f}")

        # 保存最佳模型
        if val_miou > best_miou:
            best_miou = val_miou
            best_model_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved with mIoU: {best_miou:.4f}")

        # 每5个epoch保存一次检查点
        if (epoch + 1) % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch, val_miou, checkpoint_path)


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 数据加载
    train_dataset = EnhancedWildScenesDataset('train')
    val_dataset = EnhancedWildScenesDataset('valid')

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

    # 模型定义
    model = deeplabv3_resnet50(num_classes=18)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    num_epochs = 10
    save_dir = 'model_checkpoints'
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir)

print("Training completed!")