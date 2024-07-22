import os
import sys
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from data_split import WildScenesDataset
from utils.transforms import TrainTransform, TestTransform, AdvancedAugmentation
from torchvision.transforms import functional as TF

color_map = {
    0: [224, 31, 77],  # Bush
    1: [64, 180, 78],  # Dirt
    2: [26, 127, 127],  # Fence
    3: [127, 127, 127],  # Grass
    4: [145, 24, 178],  # Gravel
    5: [125, 128, 16],  # Log
    6: [251, 225, 48],  # Mud
    7: [248, 190, 190],  # Other-object
    8: [89, 239, 239],  # Other-terrain
    9: [173, 255, 196],  # Rock
    10: [19, 0, 126],  # Sky
    11: [167, 110, 44],  # Structure
    12: [208, 245, 71],  # Tree-foliage
    13: [238, 47, 227],  # Tree-trunk
    14: [40, 127, 198],  # Water
    15: [0, 0, 0],      # 背景类（黑色）
    16: [128, 128, 128],  # 忽略类（灰色）
}

def custom_collate(batch):
        images = []
        masks = []
        for item in batch:
            image, mask = item
            
            # 确保图像大小一致（使用 224x224 或你希望的任何大小）
            image = TF.resize(image, (224, 224))
            mask = TF.resize(mask.unsqueeze(0), (224, 224), interpolation=TF.InterpolationMode.NEAREST).squeeze(0)
            
            images.append(image)
            masks.append(mask)
        
        # 堆叠图像和掩码
        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)
    
        return images, masks

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
        # assert image.shape[0] == 3, f"Image should have 3 channels, got {image.shape[0]}"
        # assert image.shape[1] == image.shape[2], f"Image should be square, got shape {image.shape}"
        # assert label.shape == image.shape[1:], f"Label shape {label.shape} doesn't match image shape {image.shape[1:]}"

        return image, label

    def _load_color_map(self):
        return {key: np.array(value) for key, value in color_map.items()}

    def _get_transform(self, dataset_type):
        if dataset_type == 'train':
            return AdvancedAugmentation()
        elif dataset_type in ['valid', 'test']:
            return TestTransform()
        else:
            raise ValueError('Invalid dataset type')

    @staticmethod
    def get_color_coded_label(label_trainId):
        """
        Convert trainId label to RGB color-coded label.
        :param label_trainId: numpy array of trainId labels
        :return: numpy array of RGB color-coded labels
        """
        height, width = label_trainId.shape
        label_RGB = np.zeros((height, width, 3), dtype=np.uint8)
        for trainId, color in color_map.items():
            label_RGB[label_trainId == trainId] = color
        return label_RGB


    def get_data_loader(dataset_type, batch_size=4):
        dataset = EnhancedWildScenesDataset(dataset_type)
        shuffle = dataset_type == 'train'
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=True,collate_fn=custom_collate)

if __name__ == '__main__':
    # 测试data_loader
    train_loader = get_data_loader('train', batch_size=4)
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}") # 一個批次的圖像數據形狀，一個批次4個圖象，每個圖象3通道，每個圖像尺寸256*341
        print(f"Batch label shape: {labels.shape}") # 一個批次的label數據形狀，單通道，表示的是trainId標注的
        print(f"Batch label unique values: {torch.unique(labels)}")
        break

    dataset = EnhancedWildScenesDataset('train')
    image, label = dataset[0]
    color_coded_label = dataset.get_color_coded_label(label.numpy())
    print(f"Color-coded label shape: {color_coded_label.shape}") # 使用顔色編碼的label圖像形狀，發現是3通道，成了！
    