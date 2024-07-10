import numpy as np
from sklearn.metrics import confusion_matrix
import logging
import os
import torch


def setup_logger(log_file):
    """设置日志器"""
    logging.basicConfig(filename=log_file, level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)


def calculate_miou(pred, target, num_classes):
    """计算 mIoU"""
    pred = pred.flatten()
    target = target.flatten()

    cm = confusion_matrix(target, pred, labels=range(num_classes))
    intersection = np.diag(cm)
    union = np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm)

    iou = np.divide(intersection, union, out=np.zeros_like(intersection, dtype=float), where=union != 0)
    valid_iou = iou[union != 0]

    return np.mean(valid_iou) if len(valid_iou) > 0 else 0.0


def save_checkpoint(model, optimizer, epoch, miou, filename):
    """保存检查点"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'miou': miou,
    }
    torch.save(checkpoint, filename)
    logging.info(f"Checkpoint saved: {filename}")