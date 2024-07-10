import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models.segmentation import deeplabv3_resnet50
from data_load import EnhancedWildScenesDataset
from tqdm import tqdm
import numpy as np
import os
import logging
from utils.log import setup_logger, calculate_miou, save_checkpoint

def train_epoch(model, dataloader, criterion, optimizer, device, num_classes):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    total_miou = 0
    num_batches = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)['out']
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        miou = calculate_miou(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)
        if not np.isnan(miou):
            total_miou += miou
            num_batches += 1

    return total_loss / len(dataloader), total_miou / num_batches if num_batches > 0 else 0.0

def validate_epoch(model, dataloader, criterion, device, num_classes):
    """验证一个 epoch"""
    model.eval()
    total_loss = 0
    total_miou = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            miou = calculate_miou(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)
            total_miou += miou

    return total_loss / len(dataloader), total_miou / len(dataloader)

def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir, num_classes):
    """训练模型"""
    best_miou = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        current_epoch = epoch + 1  # 当前 epoch 计数（从 1 开始）
        logging.info(f"Epoch {current_epoch}/{num_epochs}")

        train_loss, train_miou = train_epoch(model, train_loader, criterion, optimizer, device, num_classes)
        logging.info(f"Epoch {current_epoch} - Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}")

        val_loss, val_miou = validate_epoch(model, val_loader, criterion, device, num_classes)
        logging.info(f"Epoch {current_epoch} - Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}")

        if val_miou > best_miou:
            best_miou = val_miou
            best_model_path = os.path.join(save_dir, f'best_model_epoch_{current_epoch}.pth')
            torch.save(model.state_dict(), best_model_path)
            logging.info(f"Epoch {current_epoch} - Best model saved with mIoU: {best_miou:.4f}")

        if current_epoch % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{current_epoch}.pth')
            save_checkpoint(model, optimizer, current_epoch, val_miou, checkpoint_path)
            logging.info(f"Epoch {current_epoch} - Checkpoint saved")

    logging.info(f"Training completed after {num_epochs} epochs.")

if __name__ == "__main__":
    # 设置日志
    setup_logger('training.log')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 数据加载
    train_loader = EnhancedWildScenesDataset.get_data_loader('train', batch_size=4)
    val_loader = EnhancedWildScenesDataset.get_data_loader('valid', batch_size=4)

    # # 检查数据集
    # train_dataset = train_loader.dataset
    # val_dataset = val_loader.dataset
    #
    # # 测试
    # image, label = train_dataset[0]
    # print(f"Image shape: {image.shape}, Label shape: {label.shape}")

    # 模型定义
    num_classes = 18
    model = deeplabv3_resnet50(num_classes=num_classes)
    model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练
    num_epochs = 30
    save_dir = 'model_checkpoints'
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, save_dir, num_classes)

logging.info("Training completed!")
