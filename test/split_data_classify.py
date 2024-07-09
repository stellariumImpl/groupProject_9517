import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms
from sklearn.model_selection import TimeSeriesSplit
import multiprocessing
import matplotlib.pyplot as plt


class WildScenesDataset(Dataset):
    """
    WildScenes数据集的自定义Dataset类。
    """

    def __init__(self, root_dir, transform=None):
        """
        初始化WildScenes数据集。

        参数：
            root_dir (str): 数据集的根目录。
            transform (callable, 可选): 应用于样本的可选转换。
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_dir = os.path.join(root_dir, 'image')
        self.indexlabel_dir = os.path.join(root_dir, 'indexLabel')
        self.label_dir = os.path.join(root_dir, 'label')

        self.image_files = sorted(os.listdir(self.image_dir))
        self.timestamps = [float(f.split('.')[0].replace('-', '.')) for f in self.image_files]

        self.classes = [
            'Dirt', 'Gravel', 'Mud', 'Other-terrain', 'Bush', 'Grass', 'Log', 'Tree-foliage',
            'Tree-trunk', 'Fence', 'Other-object', 'Rock', 'Structure', 'Water', 'Sky'
        ]
        self.num_classes = len(self.classes)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 注意：这里没有显式的process_mask步骤。
        # 这是因为掩码处理隐含在__getitem__方法中，
        # 通过transform和generate_final_mask的使用来完成。
        # 然而，我们可能想要添加一个单独的process_mask方法以获得更明确的控制。

    def __len__(self):
        """
        返回数据集中样本的总数。
        """
        return len(self.image_files)

    def __getitem__(self, idx):
        """
        从数据集中检索单个样本。

        参数：
            idx (int): 要检索的样本的索引。

        返回：
            tuple: (image, final_mask)，其中image是输入图像，final_mask是生成的掩码。
        """
        img_name = self.image_files[idx]
        image_path = os.path.join(self.image_dir, img_name)
        indexlabel_path = os.path.join(self.indexlabel_dir, img_name)
        label_path = os.path.join(self.label_dir, img_name)

        image = Image.open(image_path).convert('RGB')
        indexlabel = Image.open(indexlabel_path).convert('L')
        label = Image.open(label_path).convert('L')

        if self.transform:
            image = self.transform(image)
            indexlabel = self.transform(indexlabel)
            label = self.transform(label)

        indexlabel = indexlabel.squeeze()
        label = label.squeeze()

        final_mask = self.generate_final_mask(indexlabel, label)

        return image, final_mask

    def generate_final_mask(self, indexlabel, label):
        """
        通过组合indexlabel和label生成最终掩码。

        参数：
            indexlabel (Tensor): 索引标签张量。
            label (Tensor): 二进制标签张量。

        返回：
            Tensor: 最终掩码。
        """
        final_mask = torch.where(label > 0, indexlabel, torch.zeros_like(indexlabel))
        return final_mask


def time_series_split(dataset, n_splits=5):
    """
    对数据集执行时间序列分割。

    参数：
        dataset (Dataset): 要分割的数据集。
        n_splits (int): 执行的分割次数。

    返回：
        tuple: 最后一次分割的索引（训练验证索引，测试索引）。
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(dataset.timestamps))
    return splits[-1]


def get_dataloader(dataset, indices, batch_size=4, num_workers=2, shuffle=False):
    """
    为数据集的子集创建一个DataLoader。

    参数：
        dataset (Dataset): 完整的数据集。
        indices (list): 要包含在子集中的索引。
        batch_size (int): 每批样本的数量。
        num_workers (int): 用于数据加载的子进程数。
        shuffle (bool): 是否打乱数据。

    返回：
        DataLoader: 创建的DataLoader。
    """
    subset = Subset(dataset, indices)
    return DataLoader(subset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def analyze_split(dataset, train_idx, valid_idx, test_idx):
    """
    分析数据集的分割情况。

    参数：
        dataset (Dataset): 完整的数据集。
        train_idx (list): 训练集的索引。
        valid_idx (list): 验证集的索引。
        test_idx (list): 测试集的索引。
    """
    all_timestamps = np.array(dataset.timestamps)

    train_timestamps = all_timestamps[train_idx]
    valid_timestamps = all_timestamps[valid_idx]
    test_timestamps = all_timestamps[test_idx]

    print("\nDataset Split Analysis:")
    print(f"Training set: {len(train_idx)} samples")
    print(f"Validation set: {len(valid_idx)} samples")
    print(f"Test set: {len(test_idx)} samples")

    print("\nTimestamp Range:")
    print(f"Training set: {min(train_timestamps)} to {max(train_timestamps)}")
    print(f"Validation set: {min(valid_timestamps)} to {max(valid_timestamps)}")
    print(f"Test set: {min(test_timestamps)} to {max(test_timestamps)}")

    plt.figure(figsize=(12, 6))
    plt.scatter(range(len(train_timestamps)), train_timestamps, label='Training', alpha=0.6)
    plt.scatter(range(len(train_timestamps), len(train_timestamps) + len(valid_timestamps)),
                valid_timestamps, label='Validation', alpha=0.6)
    plt.scatter(range(len(train_timestamps) + len(valid_timestamps),
                      len(train_timestamps) + len(valid_timestamps) + len(test_timestamps)),
                test_timestamps, label='Test', alpha=0.6)
    plt.xlabel('Sample Index')
    plt.ylabel('Timestamp')
    plt.title('Dataset Split Timestamp Distribution')
    plt.legend()
    plt.show()


def main():
    """
    主函数，用于执行整个数据处理和分析流程。
    """
    root_dir = '../WildScenes_Dataset-61gd5a0t-/data/WildScenes/WildScenes2d/V-01'
    batch_size = 4
    num_workers = 2

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    full_dataset = WildScenesDataset(root_dir, transform=transform)

    train_valid_idx, test_idx = time_series_split(full_dataset)

    train_size = int(0.8 * len(train_valid_idx))
    train_idx = train_valid_idx[:train_size]
    valid_idx = train_valid_idx[train_size:]

    analyze_split(full_dataset, train_idx, valid_idx, test_idx)

    train_loader = get_dataloader(full_dataset, train_idx, batch_size, num_workers, shuffle=True)
    valid_loader = get_dataloader(full_dataset, valid_idx, batch_size, num_workers)
    test_loader = get_dataloader(full_dataset, test_idx, batch_size, num_workers)

    print("\nDataloader Information:")
    print(f"Number of batches in training set: {len(train_loader)}")
    print(f"Number of batches in validation set: {len(valid_loader)}")
    print(f"Number of batches in test set: {len(test_loader)}")

    print("\nClass Information:")
    for idx, cls in enumerate(full_dataset.classes):
        print(f"{idx}: {cls}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()