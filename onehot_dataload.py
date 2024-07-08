import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn.functional as F
import multiprocessing


class WildScenesDataset(Dataset):
    def __init__(self, root_dir, transform=None, num_classes=10):
        self.root_dir = root_dir
        self.transform = transform
        self.num_classes = num_classes
        self.image_dir = os.path.join(root_dir, 'image')
        self.indexlabel_dir = os.path.join(root_dir, 'indexLabel')
        self.label_dir = os.path.join(root_dir, 'label')

        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        image_path = os.path.join(self.image_dir, img_name)
        indexlabel_path = os.path.join(self.indexlabel_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        image = Image.open(image_path).convert('RGB')
        indexlabel = Image.open(indexlabel_path)
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
            indexlabel = self.transform(indexlabel)
            label = self.transform(label)

        # Process indexLabel (convert to one-hot encoding)
        indexlabel = self.to_onehot(indexlabel)

        # Process label (convert to class indices)
        label = self.to_class_indices(label)

        return image, indexlabel, label

    def to_onehot(self, indexlabel):
        indexlabel = indexlabel.long()
        one_hot = torch.zeros(self.num_classes, *indexlabel.shape[1:])
        one_hot = one_hot.scatter_(0, indexlabel, 1)
        return one_hot

    def to_class_indices(self, label):
        return label.long().squeeze()


def get_dataloader(root_dir, batch_size=4, num_workers=2, num_classes=10):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = WildScenesDataset(root_dir, transform=transform, num_classes=num_classes)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


def main():
    root_dir = '../WildScenes_Dataset-61gd5a0t-/data/WildScenes/WildScenes2d/V-01'
    dataloader = get_dataloader(root_dir, num_classes=10)

    for batch_images, batch_indexlabels, batch_labels in dataloader:
        print(
            f"Batch shapes: Images {batch_images.shape}, IndexLabels {batch_indexlabels.shape}, Labels {batch_labels.shape}")
        print(f"IndexLabels unique values: {torch.unique(batch_indexlabels)}")
        print(f"Labels unique values: {torch.unique(batch_labels)}")
        break   # 演示所用，训练时请删除此行


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()