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
from models.custom_mask2former import CustomMask2Former
from models.dense_unet import DenseUNet, TransitionUp, DenseBlock
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F
import math


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
            # outputs = outputs['out'] if isinstance(outputs, dict) else outputs

            # if len(labels.shape) == 4 and labels.shape[1] > 1:
            #     labels = torch.argmax(labels, dim=1)

            # if outputs.shape[2:] != labels.shape[1:]:
            #     outputs = F.interpolate(outputs, size=labels.shape[1:], mode='bilinear', align_corners=True)

            # print(f"Outputs type: {type(outputs)}")
            # print(f"Outputs attributes: {dir(outputs)}")
            # print(f"Labels shape: {labels.shape}")

            # 检查outputs中可能包含分割结果的属性
            if hasattr(outputs, 'class_queries_logits'):
                segmentation_output = outputs.class_queries_logits
            elif hasattr(outputs, 'segmentation_logits'):
                segmentation_output = outputs.segmentation_logits
            elif hasattr(outputs, 'masks_queries_logits'):
                segmentation_output = outputs.masks_queries_logits
            else:
                raise ValueError(
                    f"Unable to find appropriate segmentation output. Available attributes: {dir(outputs)}")

            # print(f"Segmentation output shape: {segmentation_output.shape}")

            # 如果输出是3D的 [batch_size, num_queries, num_classes]
            if len(segmentation_output.shape) == 3:
                batch_size, num_queries, num_classes = segmentation_output.shape
                # 我们需要将这个输出转换为像素级的预测
                # 这里我们假设每个query对应图像的一个区域
                # 获取图像的高度和宽度
                height, width = labels.shape[-2:]

                # 计算最接近的正方形网格大小
                grid_size = int(math.ceil(math.sqrt(num_queries)))
                segmentation_output = segmentation_output.permute(0, 2, 1).view(batch_size, num_classes, grid_size,
                                                                                grid_size)

            # print(f"Reshaped segmentation output shape: {segmentation_output.shape}")

            # 确保输出和标签有相同的空间维度
            if segmentation_output.shape[-2:] != labels.shape[-2:]:
                segmentation_output = nn.functional.interpolate(segmentation_output, size=labels.shape[-2:],
                                                                mode='bilinear', align_corners=False)

            loss = criterion(segmentation_output, labels)
            # loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()

        total_loss += loss.item()
        pred = torch.argmax(segmentation_output, dim=1)
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
            # outputs = outputs['out'] if isinstance(outputs, dict) else outputs

            # print(f"Validation - Outputs type: {type(outputs)}")
            # print(f"Validation - Outputs attributes: {dir(outputs)}")
            # print(f"Validation - Labels shape: {labels.shape}")

            # 提取分割输出
            if hasattr(outputs, 'class_queries_logits'):
                segmentation_output = outputs.class_queries_logits
            elif hasattr(outputs, 'masks_queries_logits'):
                segmentation_output = outputs.masks_queries_logits
            else:
                raise ValueError(
                    f"Unable to find appropriate segmentation output. Available attributes: {dir(outputs)}")

            # print(f"Validation - Segmentation output shape: {segmentation_output.shape}")

            # 如果输出是3D的 [batch_size, num_queries, num_classes]
            if len(segmentation_output.shape) == 3:
                batch_size, num_queries, num_classes = segmentation_output.shape
                # 获取图像的高度和宽度
                height, width = labels.shape[-2:]

                # 计算最接近的正方形网格大小
                grid_size = int(math.ceil(math.sqrt(num_queries)))

                # 重塑输出为 [batch_size, num_classes, grid_size, grid_size]
                segmentation_output = segmentation_output.permute(0, 2, 1).view(batch_size, num_classes, grid_size,
                                                                                grid_size)

                # 调整大小以匹配标签的尺寸
                segmentation_output = F.interpolate(segmentation_output, size=(height, width), mode='bilinear',
                                                    align_corners=False)

            # print(f"Validation - Reshaped segmentation output shape: {segmentation_output.shape}")

            if len(labels.shape) == 4 and labels.shape[1] > 1:
                labels = torch.argmax(labels, dim=1)

            loss = criterion(segmentation_output, labels)

            total_loss += loss.item()
            pred = torch.argmax(segmentation_output, dim=1)
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
    # 保存检查点
    state = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'metrics': metrics
    }
    torch.save(state, filename)


def setup_logger(log_file):
    # 设置日志记录器
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)


def train(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, save_dir, num_classes):
    # 训练模型的主函数
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

        if current_epoch % 5 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{current_epoch}.pth')
            save_checkpoint(model, optimizer, current_epoch, metrics, checkpoint_path)
            logging.info(f"Epoch {current_epoch} - Checkpoint saved")

        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"Current learning rate: {current_lr:.6f}")

    logging.info(f"Training completed after {num_epochs} epochs.")
    return best_model_path


# def print_model_channels(model):
#     def hook(module, input, output):
#         print(f"{module.__class__.__name__}: Input shape: {input[0].shape}, Output shape: {output.shape}")
#
#     for name, layer in model.named_modules():
#         if isinstance(layer, (nn.Conv2d, nn.ConvTranspose2d, DenseBlock, TransitionUp)):
#             layer.register_forward_hook(hook)


if __name__ == "__main__":
    setup_logger('training.log')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_loader = EnhancedWildScenesDataset.get_data_loader('train', batch_size=8)
    val_loader = EnhancedWildScenesDataset.get_data_loader('valid', batch_size=8)

    # test_loader = EnhancedWildScenesDataset.get_data_loader('test', batch_size=8)

    num_classes = 18  # 根据您的数据集调整这个值

    # 选择模型
    # model = CustomDeepLabV3(num_classes=num_classes).to(device)
    model = CustomMask2Former(num_classes=num_classes).to(device)
    # model = DenseUNet(in_channels=3, num_classes=num_classes, pretrained=True).to(device)
    # print_model_channels(model)
    # dummy_input = torch.randn(1, 3, 256, 256).to(device)
    # _ = model(dummy_input)
    #
    # # Use a small batch size for testing
    # test_input = torch.randn(1, 3, 256, 256).to(device)
    # try:
    #     output = model(test_input)
    #     print(f"Model output shape: {output.shape}")
    # except Exception as e:
    #     print(f"Error during forward pass: {e}")

    # 选择损失函数
    # criterion = FocalLoss(alpha=1, gamma=2)
    criterion = CombinedLoss(weight_focal=1.0, weight_dice=0.5)

    # 选择优化器
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

    save_dir = 'model_checkpoints'
    best_model_path = train(model, train_loader, val_loader, criterion, optimizer, scheduler,
                            num_epochs, device, save_dir, num_classes)

    logging.info("Training and prediction completed!")
