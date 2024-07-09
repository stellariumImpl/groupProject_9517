import os
import csv
import yaml
import torch
import torchvision
from torchvision.models.segmentation import deeplabv3_resnet50
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


class WildScenesDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images_dir = os.path.join(root_dir, 'image')
        self.labels_dir = os.path.join(root_dir, 'label')
        self.index_labels_dir = os.path.join(root_dir, 'indexLabel')

        self.image_files = sorted(os.listdir(self.images_dir))

        with open(os.path.join(root_dir, 'camera_calibration.yaml'), 'r') as f:
            self.camera_calibration = yaml.safe_load(f)

        self.poses_2d = {}
        poses2d_path = os.path.join(root_dir, 'poses2d.csv')
        if os.path.exists(poses2d_path):
            with open(poses2d_path, 'r') as f:
                reader = csv.reader(f)
                headers = next(reader)  # 读取标题行
                print("CSV headers:", headers)  # 打印标题以进行调试
                for row in reader:
                    # 假设第一列是时间戳
                    timestamp = row[0]
                    self.poses_2d[timestamp] = {
                        'x': float(row[1]),
                        'y': float(row[2]),
                        'z': float(row[3]),
                        'qw': float(row[4]),
                        'qx': float(row[5]),
                        'qy': float(row[6]),
                        'qz': float(row[7])
                    }
        else:
            print(f"Warning: poses2d.csv not found at {poses2d_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, img_name.replace('.jpg', '.png'))
        index_label_path = os.path.join(self.index_labels_dir, img_name.replace('.jpg', '.txt'))

        # 加载图像
        image = Image.open(img_path).convert('RGB')

        # 加载标签（假设是语义分割掩码）
        label = Image.open(label_path).convert('L')

        # 加载索引标签（如果需要）
        with open(index_label_path, 'r') as f:
            index_label = f.read().strip()

        # 获取2D姿态数据（使用文件名的时间戳部分作为键）
        timestamp = img_name.split('.')[0]  # 假设文件名格式为 "timestamp.jpg"
        pose_2d = self.poses_2d.get(timestamp, {})

        # 应用变换
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)

        return {
            'image': image,
            'label': label,
            'index_label': index_label,
            'pose_2d': pose_2d,
            'camera_calibration': self.camera_calibration,
            'image_path': img_path
        }


# 加载预训练模型
model = deeplabv3_resnet50(pretrained=True)
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)


def preprocess_image(image):
    preprocess = transforms.Compose([
        transforms.Resize((520, 520)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    return input_batch.to(device)


def predict(model, input_batch):
    with torch.no_grad():
        output = model(input_batch)['out'][0]
    output_predictions = output.argmax(0).byte().cpu().numpy()
    return output_predictions


def visualize_output(original_image, output_predictions):
    num_classes = output_predictions.max() + 1
    color_map = np.random.randint(0, 255, size=(num_classes, 3), dtype=np.uint8)
    colored_output = color_map[output_predictions]

    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(original_image)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(colored_output)
    plt.title('Segmentation Output')
    plt.axis('off')

    plt.show()


def process_dataset(dataset, model, num_samples=5):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    for i, sample in enumerate(dataloader):
        if i >= num_samples:
            break

        image = sample['image'].squeeze().to(device)
        image_path = sample['image_path'][0]

        input_batch = preprocess_image(image)
        output_predictions = predict(model, input_batch)

        original_image = Image.open(image_path)
        visualize_output(original_image, output_predictions)


if __name__ == "__main__":
    root_dir = "../WildScenes_Dataset-61gd5a0t-/data/WildScenes/WildScenes2d/V-01"
    dataset = WildScenesDataset(root_dir)
    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample keys: {sample.keys()}")
    print(f"Image shape: {sample['image'].shape}")
    print(f"Pose 2D data: {sample['pose_2d']}")
