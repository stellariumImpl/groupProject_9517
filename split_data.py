import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt

def time_series_split(timestamps, n_splits=5):
    """
    对时间戳进行时间序列分割。

    参数：
        timestamps (list): 数据集的时间戳列表。
        n_splits (int): 分割次数。

    返回：
        tuple: 最后一次分割的索引（训练验证索引，测试索引）。
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    splits = list(tscv.split(timestamps))
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

def analyze_split(timestamps, train_idx, valid_idx, test_idx):
    """
    分析数据集的分割情况。

    参数：
        timestamps (list): 数据集的时间戳列表。
        train_idx (list): 训练集的索引。
        valid_idx (list): 验证集的索引。
        test_idx (list): 测试集的索引。
    """
    all_timestamps = np.array(timestamps)

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

def split_dataset(dataset, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """
    将数据集分割为训练集、验证集和测试集。

    参数：
        dataset: 包含 timestamps 属性的数据集对象。
        train_ratio (float): 训练集占比。
        val_ratio (float): 验证集占比。
        test_ratio (float): 测试集占比。

    返回：
        tuple: (train_loader, val_loader, test_loader)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "比例之和应为1"

    train_valid_idx, test_idx = time_series_split(dataset.timestamps)
    train_size = int(train_ratio / (train_ratio + val_ratio) * len(train_valid_idx))
    train_idx = train_valid_idx[:train_size]
    valid_idx = train_valid_idx[train_size:]

    analyze_split(dataset.timestamps, train_idx, valid_idx, test_idx)

    train_loader = get_dataloader(dataset, train_idx, shuffle=True)
    val_loader = get_dataloader(dataset, valid_idx)
    test_loader = get_dataloader(dataset, test_idx)

    return train_loader, val_loader, test_loader

# 测试函数
def test_split_dataset(dataset):
    train_loader, val_loader, test_loader = split_dataset(dataset)

    print("\nDataloader Information:")
    print(f"Number of batches in training set: {len(train_loader)}")
    print(f"Number of batches in validation set: {len(val_loader)}")
    print(f"Number of batches in test set: {len(test_loader)}")

if __name__ == "__main__":
    # 这里仅作为示例，实际使用时应该从 dataload.py 导入 WildScenesDataset
    class DummyDataset:
        def __init__(self):
            self.timestamps = [i for i in range(1000)]

    dummy_dataset = DummyDataset()
    test_split_dataset(dummy_dataset)