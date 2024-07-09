import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
import random
from torchvision import transforms
from torchvision.transforms import ColorJitter  # 导入 ColorJitter

class WildScenesDataset(Dataset):
    def __init__(self, root_dir, transform=None, mask_transform=None, num_classes=15):
        self.root_dir = root_dir
        self.transform = transform
        self.mask_transform = mask_transform
        self.num_classes = num_classes
        self.image_dir = os.path.join(root_dir, 'image')
        self.indexlabel_dir = os.path.join(root_dir, 'indexLabel')
        self.label_dir = os.path.join(root_dir, 'label')

        self.image_files = sorted(os.listdir(self.image_dir))
        self.timestamps = [float(f.split('.')[0].replace('-', '.')) for f in self.image_files]

        self.classes = [
            'Dirt', 'Gravel', 'Mud', 'Other-terrain', 'Bush', 'Grass', 'Log', 'Tree-foliage',
            'Tree-trunk', 'Fence', 'Other-object', 'Rock', 'Structure', 'Water', 'Sky'
        ]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        indexlabel_path = os.path.join(self.indexlabel_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        image = Image.open(image_path).convert('RGB')
        indexlabel = Image.open(indexlabel_path).convert('L')
        label = Image.open(label_path).convert('L')

        image, indexlabel, label = self.apply_transforms(image, indexlabel, label)

        indexlabel = indexlabel.squeeze()
        label = label.squeeze()

        indexlabel = self.process_mask(indexlabel)
        label = self.process_mask(label)

        final_mask = self.generate_final_mask(indexlabel, label)

        return image, final_mask

    def apply_transforms(self, image, indexlabel, label):
        # 随机水平翻转
        if random.random() > 0.5:
            image = TF.hflip(image)
            indexlabel = TF.hflip(indexlabel)
            label = TF.hflip(label)

        # 随机垂直翻转
        if random.random() > 0.5:
            image = TF.vflip(image)
            indexlabel = TF.vflip(indexlabel)
            label = TF.vflip(label)

        # 随机旋转
        angle = random.choice([0, 90, 180, 270])
        image = TF.rotate(image, angle)
        indexlabel = TF.rotate(indexlabel, angle)
        label = TF.rotate(label, angle)

        # 应用其他转换
        if self.transform:
            # 确保色调调整在合法范围内
            if hasattr(self.transform, 'transforms'):
                for t in self.transform.transforms:
                    if isinstance(t, ColorJitter) and t.hue is not None:
                        if isinstance(t.hue, tuple):
                            t.hue = (max(min(t.hue[0], 0.5), -0.5), max(min(t.hue[1], 0.5), -0.5))
                        else:
                            t.hue = max(min(t.hue, 0.5), -0.5)
            image = self.transform(image)

        if self.mask_transform:
            indexlabel = self.mask_transform(indexlabel)
            label = self.mask_transform(label)

        return image, indexlabel, label

    def adjust_hue(img, hue_factor):
        # 确保 hue_factor 在 -0.5 到 0.5 之间
        hue_factor = max(min(hue_factor, 0.5), -0.5)

        # 将 PIL 图像转换为 HSV 模式的 numpy 数组
        img_hsv = np.array(img.convert('HSV'))
        np_h = img_hsv[:, :, 0].astype(np.uint8)

        # 对 hue 进行调整
        np_h = (np_h.astype(np.int16) + np.int16(hue_factor * 255)) % 256

        # 更新图像的 HSV 通道并转换回 RGB 模式
        img_hsv[:, :, 0] = np_h.astype(np.uint8)
        img_rgb = Image.fromarray(img_hsv, 'HSV').convert('RGB')

        return img_rgb

    # 替换 TF.adjust_hue 函数为自定义的 adjust_hue 函数
    TF.adjust_hue = adjust_hue

    def process_mask(self, mask):
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        mask = torch.clamp(mask, 0, self.num_classes - 1)
        return mask

    def generate_final_mask(self, indexlabel, label):
        final_mask = torch.where(label > 0, indexlabel, torch.zeros_like(indexlabel))
        return final_mask

# 测试函数
def test_dataset(dataset):
    print(f"Dataset size: {len(dataset)}")
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique values in mask: {torch.unique(mask)}")

if __name__ == "__main__":
    root_dir = '../WildScenes_Dataset-61gd5a0t-/data/WildScenes/WildScenes2d/V-01'
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = WildScenesDataset(root_dir, transform=image_transform, mask_transform=mask_transform)
    test_dataset(dataset)