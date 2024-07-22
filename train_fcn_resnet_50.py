import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.cuda.amp import GradScaler, autocast
from data_load_for_mask2former import EnhancedWildScenesDataset
from tqdm import tqdm
import numpy as np
import os
import logging
from utils.metrics import calculate_miou_train, calculate_pixel_accuracy, calculate_dice_coefficient
from utils.losses import CombinedLoss
from utils.log import setup_logger, save_checkpoint
import torch.nn.functional as F
from torchvision import models
from torchvision.models.segmentation import FCN_ResNet50_Weights  # 确保正确导入
import wandb  # 导入wandb

def train_epoch(model, dataloader, criterion, optimizer, device, num_classes, scaler, accumulation_steps=2):
    model.train()
    total_loss = 0
    total_miou = 0
    total_pixel_acc = 0
    total_dice = 0
    num_batches = 0
    optimizer.zero_grad()

    for i, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images, labels = images.to(device), labels.to(device)

        with torch.cuda.amp.autocast():
            outputs = model(images)['out']
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, labels)
            loss = loss / accumulation_steps

        scaler.scale(loss).backward()

        if (i + 1) % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item() * accumulation_steps
        pred = torch.argmax(outputs, dim=1)
        miou = calculate_miou_train(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)
        pixel_acc = calculate_pixel_accuracy(pred.cpu().numpy(), labels.cpu().numpy())
        dice = calculate_dice_coefficient(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)

        if not np.isnan(miou):
            total_miou += miou
            total_pixel_acc += pixel_acc
            total_dice += dice
            num_batches += 1

    if (i + 1) % accumulation_steps != 0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    return (total_loss / len(dataloader),
            total_miou / num_batches if num_batches > 0 else 0.0,
            total_pixel_acc / num_batches if num_batches > 0 else 0.0,
            total_dice / num_batches if num_batches > 0 else 0.0)

def validate_epoch(model, dataloader, criterion, device, num_classes):
    model.eval()
    total_loss = 0
    total_miou = 0
    total_pixel_acc = 0
    total_dice = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validating"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)['out']
            outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            pred = torch.argmax(outputs, dim=1)
            miou = calculate_miou_train(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)
            pixel_acc = calculate_pixel_accuracy(pred.cpu().numpy(), labels.cpu().numpy())
            dice = calculate_dice_coefficient(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)

            total_miou += miou
            total_pixel_acc += pixel_acc
            total_dice += dice

    num_batches = len(dataloader)
    return (total_loss / num_batches,
            total_miou / num_batches,
            total_pixel_acc / num_batches,
            total_dice / num_batches)

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir, num_classes):
    # 初始化wandb
    wandb.init(
        project="wildscenes-segmentation-fcn_resnet50",  # 设置您的项目名称
        config={
            "model": "FCN-ResNet50",
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "num_classes": num_classes,
            "scheduler": "CosineAnnealingWarmRestarts",
            "loss": "CombinedLoss",
        }
    )
    
    best_miou = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scaler = GradScaler()

    for epoch in range(num_epochs):
        torch.cuda.empty_cache()
        current_epoch = epoch + 1
        logging.info(f"Epoch {current_epoch}/{num_epochs}")

        train_loss, train_miou, train_pixel_acc, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, device, num_classes, scaler)
        logging.info(f"Epoch {current_epoch} - Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, "
                     f"Train Pixel Acc: {train_pixel_acc:.4f}, Train Dice: {train_dice:.4f}")

        val_loss, val_miou, val_pixel_acc, val_dice = validate_epoch(
            model, val_loader, criterion, device, num_classes)
        logging.info(f"Epoch {current_epoch} - Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}, "
                     f"Val Pixel Acc: {val_pixel_acc:.4f}, Val Dice: {val_dice:.4f}")

        scheduler.step()

         # 记录指标到wandb
        wandb.log({
            "epoch": current_epoch,
            "train_loss": train_loss,
            "train_miou": train_miou,
            "train_pixel_acc": train_pixel_acc,
            "train_dice": train_dice,
            "val_loss": val_loss,
            "val_miou": val_miou,
            "val_pixel_acc": val_pixel_acc,
            "val_dice": val_dice,
            "learning_rate": optimizer.param_groups[0]['lr']
        })

        metrics = {
            'miou': val_miou,
            'pixel_acc': val_pixel_acc,
            'dice': val_dice
        }

        if val_miou > best_miou:
            best_miou = val_miou
            best_model_path = os.path.join(save_dir, f'best_model_epoch_{current_epoch}.pth')
            save_checkpoint(model, optimizer, current_epoch, metrics, best_model_path)
            logging.info(f"Epoch {current_epoch} - Best model saved with mIoU: {best_miou:.4f}")

        # if current_epoch % 5 == 0:
        #     checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{current_epoch}.pth')
        #     save_checkpoint(model, optimizer, current_epoch, metrics, checkpoint_path)
        #     logging.info(f"Epoch {current_epoch} - Checkpoint saved")

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current learning rate: {current_lr:.6f}")

    logging.info(f"Training completed after {num_epochs} epochs.")
    wandb.finish()  # 结束wandb运行
    return best_model_path

if __name__ == "__main__":
    save_dir = os.path.join('model_checkpoints', 'FCN')
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, 'training.log')
    setup_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 修改数据加载器以使用新的增强
    train_loader = EnhancedWildScenesDataset.get_data_loader('train', batch_size=16)
    val_loader = EnhancedWildScenesDataset.get_data_loader('valid', batch_size=16)

    num_classes = 17

    # 加载预训练的FCN模型
    model = models.segmentation.fcn_resnet50(weights=FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1)
    model.classifier[4] = nn.Conv2d(512, num_classes, kernel_size=1)
    model = model.to(device)

    criterion = CombinedLoss(weight_focal=0.75, weight_dice=0.25)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    num_epochs = 60
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)

    best_model_path = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                            num_epochs, device, save_dir, num_classes)

    logging.info("Training and prediction completed!")
