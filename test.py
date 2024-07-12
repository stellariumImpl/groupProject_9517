import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import os
import logging
from data_split import WildScenesDataset
from utils.transforms import PairNormalizeToTensor, PairResize, PairCrop
from models.custom_deeplabv3 import CustomDeepLabV3
from utils.metrics import calculate_miou, calculate_confusion_matrix, calculate_dice_coefficient
import torch.nn as nn
from collections import OrderedDict
from data_load import EnhancedWildScenesDataset
from torchvision.models.segmentation import deeplabv3_resnet101


def is_pil_image(img):
    return isinstance(img, Image.Image)


def is_numpy_array(arr):
    return isinstance(arr, np.ndarray)


def numpy_to_pil(image):
    """Convert a numpy array to PIL Image."""
    if is_pil_image(image):
        return image
    if image.dtype == np.float32:
        image = (image * 255).astype(np.uint8)
    elif image.dtype == np.int64:
        image = image.astype(np.uint8)

    if image.ndim == 2:
        return Image.fromarray(image, mode='L')
    elif image.ndim == 3:
        return Image.fromarray(image, mode='RGB')
    else:
        raise ValueError(f"Unsupported number of dimensions: {image.ndim}")


def test(net, data, device, resize_to=256, n_class=8, compare=False, output_dir='image_split'):
    net.to(device)
    net.eval()
    total_cm = np.zeros((n_class, n_class))
    total_batch_miou = 0.
    total_batch_dice = 0.

    offset = 690
    pair_crop = PairCrop(offsets=(offset, None))
    pair_resize = PairResize(size=resize_to)
    pair_norm_to_tensor = PairNormalizeToTensor(norm=True)

    with torch.no_grad():
        bar_format = '{desc}{postfix}|{n_fmt}/{total_fmt}|{percentage:3.0f}%|{bar}|{elapsed}<{remaining}'
        tqdm_data = tqdm(data, ncols=120, bar_format=bar_format, desc='Test')
        for i_batch, (im, lb) in enumerate(tqdm_data, start=1):
            if is_pil_image(im):
                logging.debug(f"Image size: {im.size}, mode: {im.mode}")
            elif is_numpy_array(im):
                logging.debug(f"Image shape: {im.shape}, dtype: {im.dtype}")
            else:
                logging.error(f"Unexpected image type: {type(im)}")
                continue

            if is_pil_image(lb):
                logging.debug(f"Label size: {lb.size}, mode: {lb.mode}")
            elif is_numpy_array(lb):
                logging.debug(f"Label shape: {lb.shape}, dtype: {lb.dtype}")
            else:
                logging.error(f"Unexpected label type: {type(lb)}")
                continue

            try:
                im = numpy_to_pil(im)
                lb = numpy_to_pil(lb)

                im_t, lb_t = pair_crop(im, lb)
                im_t, lb_t = pair_resize(im_t, lb_t)
                im_t, lb_t = pair_norm_to_tensor(im_t, lb_t)
            except Exception as e:
                logging.error(f"Error in data preprocessing: {str(e)}")
                continue

            im_t = im_t.to(device)
            im_t = im_t.unsqueeze(0)
            output = net(im_t)

            if isinstance(output, OrderedDict):
                logging.debug(f"Model output is OrderedDict with keys: {output.keys()}")
                if 'out' in output:
                    output = output['out']
                else:
                    output = next(iter(output.values()))
                logging.debug(f"Using output tensor with shape: {output.shape}")
            elif isinstance(output, torch.Tensor):
                logging.debug(f"Model output is Tensor with shape: {output.shape}")
            else:
                logging.error(f"Unexpected output type: {type(output)}")
                continue

            pred = torch.argmax(F.softmax(output, dim=1), dim=1)

            pred = pred.unsqueeze(1)
            pred = pred.type(torch.float)
            pred = F.interpolate(pred, size=(lb.size[1] - offset, lb.size[0]), mode='nearest')
            pred = pred.type(torch.uint8)
            pred = pred.squeeze(0).squeeze(0)
            pred = pred.cpu().numpy()

            supplement = np.zeros((offset, lb.size[0]), dtype=np.uint8)
            pred = np.append(supplement, pred, axis=0)

            lb_np = np.array(lb)

            batch_cm = calculate_confusion_matrix(pred, lb_np, n_class)
            total_cm += batch_cm

            batch_miou = calculate_miou(batch_cm)
            batch_dice = calculate_dice_coefficient(pred, lb_np, n_class)
            total_batch_miou += batch_miou
            total_batch_dice += batch_dice

            if compare:
                os.makedirs(output_dir, exist_ok=True)
                save_comparison_image(im, lb, pred, batch_cm, i_batch, output_dir)

            tqdm_str = 'mIoU={:.4f}|Dice={:.4f}'
            tqdm_data.set_postfix_str(
                tqdm_str.format(batch_miou, batch_dice))

        mean_iou = calculate_miou(total_cm)
        mean_dice = total_batch_dice / len(data)
        total_batch_miou /= len(data)

        logging.info(f'Test mIoU: {mean_iou:.4f} | Test Dice: {mean_dice:.4f}')
        return mean_iou, mean_dice


def save_comparison_image(im, lb, pred, batch_cm, index, output_dir):
    fontsize = 16
    fig, ax = plt.subplots(2, 2, figsize=(20, 15))
    ax = ax.flatten()

    # 输入图像
    ax[0].imshow(im)
    ax[0].set_title('Input Image', fontsize=fontsize)

    # 真实标签
    lb_np = np.array(lb)
    lb_color = EnhancedWildScenesDataset.get_color_coded_label(lb_np)
    ax[1].imshow(lb_color)
    ax[1].set_title('Ground Truth', fontsize=fontsize)

    # 计算和显示 mIoU
    batch_miou = calculate_miou(batch_cm)
    fig.suptitle(f'mIoU:{batch_miou:.4f}', fontsize=fontsize)

    # 预测结果
    pred_color = EnhancedWildScenesDataset.get_color_coded_label(pred)
    ax[2].imshow(pred_color)
    ax[2].set_title('Prediction', fontsize=fontsize)

    # 预测结果叠加在原图上
    mask = (pred != 0).astype(np.uint8) * 255
    mask = mask[..., np.newaxis]
    pred_rgba = np.concatenate([pred_color, mask], axis=2)

    im_pil = Image.fromarray(np.array(im)).convert('RGBA')
    pred_pil = Image.fromarray(pred_rgba).convert('RGBA')
    im_comp = Image.alpha_composite(im_pil, pred_pil)

    ax[3].imshow(im_comp)
    ax[3].set_title('Prediction over Input', fontsize=fontsize)

    # 调整子图布局
    plt.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99,
                        wspace=0.01, hspace=0.01)

    # 保存图像
    plt.savefig(f'{output_dir}/comparison_image_{index}.png')
    plt.close(fig)


def get_model(model_name, in_channels, num_classes, device, load_weight=None, adjust_classifier=False):
    """
    获取并初始化模型

    Args:
    model_name (str): 模型名称，例如 'custom_deeplabv3'
    in_channels (int): 输入通道数
    num_classes (int): 类别数量
    device (torch.device): 使用的设备（CPU 或 GPU）
    load_weight (str): 预训练权重文件路径，如果为None则不加载
    adjust_classifier (bool): 是否调整分类器以匹配新的类别数量

    Returns:
    torch.nn.Module: 初始化好的模型
    """
    if model_name.lower() == 'custom_deeplabv3':
        model = CustomDeepLabV3(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unsupported model name: {model_name}")

    if load_weight:
        try:
            checkpoint = torch.load(load_weight, map_location=device)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            # 检查分类器的权重形状
            classifier_weight = state_dict['deeplabv3.classifier.4.4.weight']
            classifier_bias = state_dict['deeplabv3.classifier.4.4.bias']
            loaded_num_classes = classifier_weight.size(0)

            if loaded_num_classes != num_classes:
                if adjust_classifier:
                    logging.info(f"Adjusting classifier from {loaded_num_classes} to {num_classes} classes.")
                    # 调整分类器权重
                    new_classifier_weight = nn.Parameter(torch.randn(num_classes, classifier_weight.size(1), 1, 1))
                    new_classifier_bias = nn.Parameter(torch.randn(num_classes))

                    # 复制共同的类别
                    min_classes = min(loaded_num_classes, num_classes)
                    new_classifier_weight.data[:min_classes] = classifier_weight.data[:min_classes]
                    new_classifier_bias.data[:min_classes] = classifier_bias.data[:min_classes]

                    state_dict['deeplabv3.classifier.4.4.weight'] = new_classifier_weight
                    state_dict['deeplabv3.classifier.4.4.bias'] = new_classifier_bias
                else:
                    raise ValueError(
                        f"Model checkpoint has {loaded_num_classes} classes, but current model has {num_classes} classes. Set adjust_classifier=True to adjust the classifier.")

            model.load_state_dict(state_dict)
            logging.info(f"Loaded model weights from {load_weight}")
            if 'epoch' in checkpoint:
                logging.info(f"Model checkpoint from epoch {checkpoint['epoch']}")
            if 'metrics' in checkpoint:
                logging.info(f"Model metrics: mIoU: {checkpoint['metrics']['miou']:.4f}, "
                             f"Pixel Accuracy: {checkpoint['metrics']['pixel_acc']:.4f}, "
                             f"Dice Coefficient: {checkpoint['metrics']['dice']:.4f}")
        except Exception as e:
            logging.error(f"Error loading model weights: {str(e)}")
            logging.info("Initializing model with random weights.")

    return model


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    load_file = 'model_checkpoints/best_model_epoch_47.pth'

    num_classes = 18  # 使用与预训练模型相同的类别数
    adjust_classifier = False

    try:
        mod = get_model('custom_deeplabv3', in_channels=3, num_classes=num_classes,
                        device=dev, load_weight=load_file, adjust_classifier=adjust_classifier)
        logging.info("Model loaded successfully")
    except Exception as e:
        logging.error(f"Failed to load model: {str(e)}")
        exit(1)

    test_dataset = WildScenesDataset('test')
    mean_iou, mean_dice = test(net=mod,
                               data=test_dataset,
                               resize_to=578,
                               n_class=num_classes,
                               device=dev,
                               compare=True)

    logging.info(f"Final Mean IoU: {mean_iou:.4f}")
    logging.info(f"Final Mean Dice Coefficient: {mean_dice:.4f}")
