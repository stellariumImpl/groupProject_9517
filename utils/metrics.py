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
    return np.mean(ious)  #  Returns the mean, which is a scalar

def calculate_pixel_accuracy(pred, target):
    correct = (pred == target).sum()
    total = pred.size
    return correct / total  # This is already a scalar



def calculate_confusion_matrix(pred, target, num_classes):
    mask = (target >= 0) & (target < num_classes)
    return np.bincount(
        num_classes * target[mask].astype(int) + pred[mask],
        minlength=num_classes ** 2).reshape(num_classes, num_classes)


def calculate_miou(pred, target, num_classes):
    ious = []
    pred = pred.ravel()
    target = target.ravel()
    for cls in range(num_classes):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = np.logical_and(pred_inds, target_inds).sum()
        union = np.logical_or(pred_inds, target_inds).sum()
        if union == 0:
            ious.append(float('nan'))  
        else:
            ious.append(intersection / union)
    miou = np.nanmean(ious) 
    return miou, np.array(ious)


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
