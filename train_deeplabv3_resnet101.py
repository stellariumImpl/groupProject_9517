import torch
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.cuda.amp import GradScaler, autocast
from data_load import EnhancedWildScenesDataset
from tqdm import tqdm
import numpy as np
import os
import logging
from utils.metrics import calculate_miou_train, calculate_pixel_accuracy, calculate_dice_coefficient
from utils.losses import CombinedLoss
from models.custom_deeplabv3 import CustomDeepLabV3
import torch.nn.functional as F
import wandb  

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


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

        with torch.cuda.amp.autocast():
            outputs = model(images)
            outputs = outputs['out'] if isinstance(outputs, dict) else outputs

            if len(labels.shape) == 4 and labels.shape[1] > 1:
                labels = torch.argmax(labels, dim=1)

            if outputs.shape[2:] != labels.shape[1:]:
                outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=True)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

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
            outputs = outputs['out'] if isinstance(outputs, dict) else outputs

            if len(labels.shape) == 4 and labels.shape[1] > 1:
                labels = torch.argmax(labels, dim=1)

            if outputs.shape[2:] != labels.shape[1:]:
                outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=True)

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


def save_checkpoint(model, optimizer, epoch, metrics, filename):
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(state, filename)


def setup_logger(log_file):
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir, num_classes):
    wandb.init(
        project="wildscenes-segmentation-deeplabv3_resnet101",  
        config={
            "model": "CustomDeepLabV3",
            "learning_rate": optimizer.param_groups[0]['lr'],
            "epochs": num_epochs,
            "batch_size": train_loader.batch_size,
            "num_classes": num_classes,
            "scheduler": "OneCycleLR",
            "loss": "CombinedLoss",
        }
    )
    
    best_miou = 0
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    scaler = GradScaler()

    for epoch in range(num_epochs):
        current_epoch = epoch + 1
        logging.info(f"Epoch {current_epoch}/{num_epochs}")

        train_loss, train_miou, train_pixel_acc, train_dice = train_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, num_classes, scaler)
        logging.info(f"Epoch {current_epoch} - Train Loss: {train_loss:.4f}, Train mIoU: {train_miou:.4f}, "
                     f"Train Pixel Acc: {train_pixel_acc:.4f}, Train Dice: {train_dice:.4f}")

        val_loss, val_miou, val_pixel_acc, val_dice = validate_epoch(
            model, val_loader, criterion, device, num_classes)
        logging.info(f"Epoch {current_epoch} - Val Loss: {val_loss:.4f}, Val mIoU: {val_miou:.4f}, "
                     f"Val Pixel Acc: {val_pixel_acc:.4f}, Val Dice: {val_dice:.4f}")



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
    wandb.finish() 
    return best_model_path


if __name__ == "__main__":
    save_dir = os.path.join('model_checkpoints', 'DeepLabV3 ResNet 101')
    os.makedirs(save_dir, exist_ok=True)

    log_file = os.path.join(save_dir, 'training.log')
    setup_logger(log_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader = EnhancedWildScenesDataset.get_data_loader('train', batch_size=16)
    val_loader = EnhancedWildScenesDataset.get_data_loader('valid', batch_size=16)

    # test_loader = EnhancedWildScenesDataset.get_data_loader('test', batch_size=8)

    num_classes = 17

    model = CustomDeepLabV3(num_classes=num_classes).to(device)

    # criterion = FocalLoss(alpha=1, gamma=2)
    criterion = CombinedLoss(weight_focal=1.0, weight_dice=0.5)

    # optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=0.0001, nesterov=True)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    num_epochs = 60
    steps_per_epoch = len(train_loader)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.01,
        steps_per_epoch=steps_per_epoch,
        epochs=num_epochs,
        pct_start=0.3,
        anneal_strategy='cos',
        div_factor=25,
        final_div_factor=1000
    )

    best_model_path = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                            num_epochs, device, save_dir, num_classes)

    logging.info("Training and prediction completed!")
