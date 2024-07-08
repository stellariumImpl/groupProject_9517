import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class WildScenesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'image')
        self.indexLabel_dir = os.path.join(root_dir, 'indexLabel')
        self.label_dir = os.path.join(root_dir, 'label')

        self.image_files = sorted(os.listdir(self.image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        indexLabel_path = os.path.join(self.indexLabel_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        image = Image.open(image_path).convert('RGB')
        indexLabel = Image.open(indexLabel_path)
        label = Image.open(label_path)

        if self.transform:
            image = self.transform(image)
            indexLabel = self.transform(indexLabel)
            label = self.transform(label)

        return image, indexLabel, label


def dataloader_instance(root_dir, batch_size=4, num_workers=2):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = WildScenesDataset(root_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return dataloader


if __name__ == '__main__':
    root_dir = '../WildScenes_Dataset-61gd5a0t-/data/WildScenes/WildScenes2d/V-01'
    dataloader = dataloader_instance(root_dir)

    # 遍历DataLoader，访问batch数据
    for batch_images, batch_indexLabels, batch_labels in dataloader:
        print(
            f"Batch shapes: Images {batch_images.shape}, IndexLabels {batch_indexLabels.shape}, Labels {batch_labels.shape}")
        break  # 演示所用，训练时请删除此行
