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