import os
import torch
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from torchvision import transforms
from data_load import WildScenesDataset
from split_data import split_dataset
from model import create_model, train_and_validate
from torch.cuda.amp import GradScaler, autocast

def main():
    # 检查 CUDA 是否可用
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"使用 GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU 内存总量: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        device = torch.device("cpu")
        print("CUDA 不可用，使用 CPU")

    # 定义数据集根目录
    root_dir = '../WildScenes_Dataset-61gd5a0t-/data/WildScenes/WildScenes2d/V-01'

    # 确保根目录存在
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"数据集目录 {root_dir} 不存在。请检查路径是否正确。")

    # 定义图像转换
    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 定义掩码转换
    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    # 创建数据集
    dataset = WildScenesDataset(root_dir, transform=image_transform, mask_transform=mask_transform)

    # 分割数据集
    train_loader, val_loader, test_loader = split_dataset(dataset)

    # 创建模型
    num_classes = len(dataset.classes)
    model = create_model(num_classes).to(device)

    # 定义损失函数和优化器
    criterion = CrossEntropyLoss().to(device)
    optimizer = Adam(model.parameters(), lr=0.001)

    # 创建 GradScaler 用于自动混合精度训练
    scaler = GradScaler()

    # 训练模型
    num_epochs = 10  # 可以根据需要调整

    def train_function(model, train_loader, criterion, optimizer, device):
        model.train()
        total_loss = 0

        for images, masks in train_loader:
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()

            # 使用自动混合精度
            with autocast():
                outputs = model(images)['out']
                loss = criterion(outputs, masks)

            # 使用 scaler 来调整反向传播
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    train_and_validate(model, train_loader, val_loader, num_epochs, criterion, optimizer, device)

    # 保存模型
    torch.save(model.state_dict(), 'deeplabv3_resnet50_wildscenes.pth')
    print("模型保存成功！")

    # 在测试集上评估模型
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += masks.numel()
            correct += predicted.eq(masks).sum().item()

    print(f"测试损失: {test_loss/len(test_loader):.4f}")
    print(f"测试准确率: {100.*correct/total:.2f}%")

if __name__ == "__main__":
    main()