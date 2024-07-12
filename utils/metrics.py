import numpy as np
from sklearn.metrics import confusion_matrix


def calculate_miou_train(pred, target, num_classes):
    ious = np.zeros(num_classes)
    for class_id in range(num_classes):
        pred_class = (pred == class_id)
        target_class = (target == class_id)
        intersection = np.logical_and(pred_class, target_class).sum()
        union = np.logical_or(pred_class, target_class).sum()
        if union == 0:
            ious[class_id] = 1.0 if intersection == 0 else 0.0
        else:
            ious[class_id] = intersection / union
    return np.mean(ious)  # 返回平均值，这是一个标量

def calculate_pixel_accuracy(pred, target):
    correct = (pred == target).sum()
    total = pred.size
    return correct / total  # 这已经是一个标量


# def calculate_dice_coefficient(pred, target, num_classes):
#     dice_scores = np.zeros(num_classes)
#     for class_id in range(num_classes):
#         pred_class = (pred == class_id).astype(float)
#         target_class = (target == class_id).astype(float)
#         intersection = np.sum(pred_class * target_class)
#         pred_sum = np.sum(pred_class)
#         target_sum = np.sum(target_class)
#
#         denominator = pred_sum + target_sum
#         if denominator == 0:
#             dice_scores[class_id] = 1.0 if intersection == 0 else 0.0
#         else:
#             dice_scores[class_id] = (2 * intersection) / denominator
#
#     return np.mean(dice_scores)  # 返回平均值，这是一个标量


def calculate_confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    return np.bincount(
        num_classes * target[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2).reshape(num_classes, num_classes)


def calculate_miou(confusion_matrix):
    intersection = np.diag(confusion_matrix)
    ground_truth_set = confusion_matrix.sum(axis=1)
    predicted_set = confusion_matrix.sum(axis=0)
    union = ground_truth_set + predicted_set - intersection

    iou = np.zeros_like(intersection, dtype=float)
    for i in range(len(intersection)):
        if union[i] != 0:
            iou[i] = intersection[i] / union[i]
        else:
            iou[i] = 0.0  # 或者设置为 1.0，取决于您如何处理这种情况

    # 移除 NaN 值（如果有的话）
    valid_iou = iou[~np.isnan(iou)]

    if len(valid_iou) == 0:
        logging.warning("All IoU values are NaN!")
        return 0.0

    return np.mean(valid_iou)


def calculate_dice_coefficient(pred, target, num_classes):
    dice_scores = np.zeros(num_classes)
    for class_id in range(num_classes):
        pred_class = (pred == class_id).astype(float)
        target_class = (target == class_id).astype(float)
        intersection = np.sum(pred_class * target_class)
        union = np.sum(pred_class) + np.sum(target_class)

        if union == 0:
            dice_scores[class_id] = 1.0 if intersection == 0 else 0.0
        else:
            dice_scores[class_id] = (2. * intersection) / union

    return np.mean(dice_scores)