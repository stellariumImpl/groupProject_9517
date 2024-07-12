# data_loader/enhanced_dataset.py

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from data_split import WildScenesDataset
from utils.transforms import TrainTransform, TestTransform
from PIL import Image

color_map = {
    1: [224, 31, 77],  # Bush
    2: [64, 180, 78],  # Dirt
    3: [26, 127, 127], # Fence
    4: [127, 127, 127],# Grass
    5: [145, 24, 178], # Gravel
    6: [125, 128, 16], # Log
    7: [251, 225, 48], # Mud
    8: [248, 190, 190],# Other-object
    9: [89, 239, 239], # Other-terrain
    11: [173, 255, 196],# Rock
    12: [19, 0, 126],  # Sky
    13: [167, 110, 44],# Structure
    14: [208, 245, 71],# Tree-foliage
    15: [238, 47, 227],# Tree-trunk
    17: [40, 127, 198] # Water
}

class EnhancedWildScenesDataset(WildScenesDataset):
    def __init__(self, dataset_type, transform=None):
        super().__init__(dataset_type, transform)
        self.color_map = self._load_color_map()
        self.transform = self._get_transform(dataset_type)

    def __getitem__(self, index):
        image_path = self._data_frame['image'][index]
        label_path = self._data_frame['label'][index]

        image = Image.open(image_path).convert('RGB')
        # Use test_label_mapping to get both original and mapped labels
        original_label, mapped_label = self.test_label_mapping(label_path)

        # Convert numpy array to PIL Image for compatibility with transforms
        label = Image.fromarray(mapped_label.astype(np.uint8))

        if self.transform is not None:
            image, label = self.transform(image, label)

        # Verify shapes
        assert image.shape[0] == 3, f"Image should have 3 channels, got {image.shape[0]}"
        assert image.shape[1] == image.shape[2], f"Image should be square, got shape {image.shape}"
        assert label.shape == image.shape[
                              1:], f"Label shape {label.shape} doesn't match image shape {image.shape[1:]}"

        return image, label

    def _load_color_map(self):
        # Assume class_dict.csv is in the same directory as this script
        # csv_path = os.path.join(os.path.dirname(__file__), 'class_dict.csv')
        # df = pd.read_csv(csv_path)
        # color_map = {row['traidId']: np.array([row['r'], row['g'], row['b']]) for _, row in df.iterrows()}
        color_map = {key: np.array(value) for key, value in EnhancedWildScenesDataset.color_map.items()}
        return color_map

    def _get_transform(self, dataset_type):
        if dataset_type == 'train':
            return TrainTransform()
        elif dataset_type in ['valid', 'test']:
            return TestTransform()
        else:
            raise ValueError('Invalid dataset type')

    color_map = color_map

    @staticmethod
    def get_color_coded_label(label_trainId):
        """
        Convert trainId label to RGB color-coded label.

        :param label_trainId: numpy array of trainId labels
        :return: numpy array of RGB color-coded labels
        """
        height, width = label_trainId.shape
        label_RGB = np.zeros((height, width, 3), dtype=np.uint8)

        for trainId, color in EnhancedWildScenesDataset.color_map.items():
            label_RGB[label_trainId == trainId] = color

        return label_RGB

    # 为了测试对测试集图片分割标注，传入trainId标注的灰度图
    # def trainId2RGB(label):
    #     label_trainId = np.asarray(label)
    #     h, w = label_trainId.shape
    #     label_RGB = np.zeros((h, w, 3), dtype=np.uint8)
    #     label_RGB[label_trainId == 1] = np.array([224, 31, 77])
    #     ...
    #     return label_RGB

    @staticmethod
    def get_data_loader(dataset_type, batch_size=4):
        dataset = EnhancedWildScenesDataset(dataset_type)
        shuffle = dataset_type == 'train'
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, pin_memory=True,
                          drop_last=True)


if __name__ == '__main__':
    train_loader = EnhancedWildScenesDataset.get_data_loader('train', batch_size=4)
    # 测试用
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")
        print(f"Batch label shape: {labels.shape}")
        break

    dataset = EnhancedWildScenesDataset('train')
    image, label = dataset[0]
    color_coded_label = dataset.get_color_coded_label(label.numpy())
    print(f"Color-coded label shape: {color_coded_label.shape}")
