import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np
import os
import logging
from data_load import EnhancedWildScenesDataset
from models.unet import UNet
from utils.metrics import calculate_miou_train, calculate_pixel_accuracy, calculate_dice_coefficient
from utils.losses import CombinedLoss
from utils.log import setup_logger, save_checkpoint
import torch.nn.functional as F
import wandb 
# This draws on the Ascetics/Pytorch-SegToolbox framework to split the training and validation processes into separate epoch functions, but the specific implementation details are original

def setup_logger(log_file):
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[
                            logging.FileHandler(log_file),
                            logging.StreamHandler()
                        ])

def save_checkpoint(model, optimizer, epoch, metrics, path):
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(state, path)

def train_epoch(model, dataloader, criterion, optimizer, scheduler, device, num_classes, scaler):
    model.train()
    total_loss = 0
    total_miou = 0
    total_pixel_acc = 0
    total_dice = 0
    num_batches = 0

    for images, labels in tqdm(dataloader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            if outputs.shape[-2:] != labels.shape[-2:]:
                outputs = F.interpolate(outputs, size=labels.shape[-2:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        pred = torch.argmax(outputs, dim=1)
        miou = calculate_miou_train(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)
        pixel_acc = calculate_pixel_accuracy(pred.cpu().numpy(), labels.cpu().numpy())
        dice = calculate_dice_coefficient(pred.cpu().numpy(), labels.cpu().numpy(), num_classes)

        if not np.isnan(miou):
            total_miou += miou
            total_pixel_acc += pixel_acc
            total_dice += dice
            num_batches += 1

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
            outputs = model(images)
            if outputs.shape[-2:] != labels.shape[-2:]:
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

def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir):
    wandb.init(
        project="wildscenes-segmentation-unet",  
        config={
            "model": "UNet",
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "num_classes": num_classes,
            "scheduler": "ReduceLROnPlateau",
            "loss": "CombinedLoss",
        }
    )
    
    best_loss = float('inf')
    best_miou = 0
    scaler = GradScaler()

    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1}/{num_epochs}")

        train_loss, train_miou, train_pixel_acc, train_dice = train_epoch(model, train_loader, criterion, optimizer, scheduler, device, num_classes, scaler)
        logging.info(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, "
                     f"Train Pixel Acc: {train_pixel_acc:.4f}, Train Dice: {train_dice:.4f}")

        val_loss, val_miou, val_pixel_acc, val_dice = validate_epoch(model, val_loader, criterion, device, num_classes)
        logging.info(f"Epoch {epoch + 1} - Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}, "
                     f"Val Pixel Acc: {val_pixel_acc:.4f}, Val Dice: {val_dice:.4f}")

        scheduler.step(val_loss)


        wandb.log({
            "epoch": epoch + 1,
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
            'val_loss': val_loss,
            'val_miou': val_miou,
            'val_pixel_acc': val_pixel_acc,
            'val_dice': val_dice
        }

        if val_miou > best_miou:
            best_miou = val_miou
            best_model_path = os.path.join(save_dir, f'best_model_epoch_{epoch + 1}.pth')
            save_checkpoint(model, optimizer, epoch + 1, metrics, best_model_path)
            logging.info(f"Epoch {epoch + 1} - Best model saved with Val mIoU: {best_miou:.4f}")

        # if (epoch + 1) % 5 == 0:
        #     checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch + 1}.pth')
        #     save_checkpoint(model, optimizer, epoch + 1, metrics, checkpoint_path)
        #     logging.info(f"Epoch {epoch + 1} - Checkpoint saved")

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current learning rate: {current_lr:.6f}")

    logging.info(f"Training completed after {num_epochs} epochs.")
    wandb.finish() 
    return best_model_path

if __name__ == "__main__":
    save_dir = 'model_checkpoints/Unet'
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, 'training.log')
    setup_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader = EnhancedWildScenesDataset.get_data_loader('train', batch_size=16)
    val_loader = EnhancedWildScenesDataset.get_data_loader('valid', batch_size=16)

    num_classes = 17

    model = UNet(3, num_classes).to(device)

    criterion = CombinedLoss(weight_focal=1.0, weight_dice=0.5)

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5, weight_decay=0.01)

    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    num_epochs = 50
    best_model_path = train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir)
    logging.info(f"Training completed! Best model saved at {best_model_path}")
