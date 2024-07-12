# transforms.py

import numpy as np
import random
import torch
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

class PairCrop:
    def __init__(self, offsets):
        self.offsets = offsets

    def __call__(self, image, label):
        left = self.offsets[1] if self.offsets[1] is not None else 0
        top = self.offsets[0] if self.offsets[0] is not None else 0
        right = image.width
        bottom = image.height
        image = image.crop((left, top, right, bottom))
        label = label.crop((left, top, right, bottom))
        return image, label

class PairResize:
    def __init__(self, size):
        self.size = size

    def __call__(self, image, label):
        image = image.resize((self.size, self.size), Image.BILINEAR)
        label = label.resize((self.size, self.size), Image.NEAREST)
        return image, label

class PairRandomHFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, label):
        if random.random() < self.p:
            image = TF.hflip(image)
            label = TF.hflip(label)
        return image, label

class AdjustColor:
    def __call__(self, image):
        image = TF.adjust_brightness(image, random.uniform(0.8, 1.2))
        image = TF.adjust_contrast(image, random.uniform(0.8, 1.2))
        image = TF.adjust_saturation(image, random.uniform(0.8, 1.2))
        return image


class NormalizeToTensor:
    def __call__(self, image, label):
        # 确保 image 是 PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)

        # 将 image 转换为张量并标准化
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # 确保 label 是 numpy 数组，然后转换为张量
        if isinstance(label, Image.Image):
            label = np.array(label)
        label = torch.from_numpy(label).long()  # 使用 long() 确保标签是长整型

        return image, label

class RandomCutout:
    def __init__(self, p=0.5, size=64):
        self.p = p
        self.size = size

    def __call__(self, image):
        if random.random() < self.p:
            h, w = image.shape[1:]
            mask = torch.ones(h, w)
            y = random.randint(0, h - self.size)
            x = random.randint(0, w - self.size)
            mask[y:y+self.size, x:x+self.size] = 0
            mask = mask.expand_as(image)
            image *= mask
        return image

class PairNormalizeToTensor(object):
    def __init__(self, norm=True, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        """
        IMAGE_NORM_MEAN = [0.485, 0.456, 0.406]  # ImageNet统计的RGB mean
        IMAGE_NORM_STD = [0.229, 0.224, 0.225]  # ImageNet统计的RGB std
        LABEL_NORM_MEAN = [0.5]  # ImageNet统计的GRAY mean
        LABEL_NORM_STD = [0.5]  # ImageNet统计的GRAY std
        :param norm: 是否正则化，默认是
        :param mean: 正则化的平均值mean
        :param std: 正则化的标准差std
        """
        super(PairNormalizeToTensor, self).__init__()
        self.norm = norm
        self.mean = mean
        self.std = std
        pass

    def __call__(self, image, label=None):
        """
        归一化，只对image除以255，label不动
        :param image: [H,W,C] PIL Image RGB 0~255
        :param label: [H,W] PIL Image trainId
        :return: [C,H,W] tensor RGB -1.0~0.0,  [H,W] tensor trainId
        """
        # torchvision.transform的API，对PIL Image类型image归一化，也就是除以255
        # 并转为tensor，维度变为[C,H,W]
        # image [C,H,W]tensor RGB 0.0~1.0
        image = TF.to_tensor(image)

        # 正则化，x=(x-mean)/std
        # 只对image正则化, image [C,H,W]tensor RGB -1.0~1.0
        if self.norm:
            image = TF.normalize(image, self.mean, self.std)

        # 先转为ndarray，再转为tensor，不归一化，维度保持不变
        # label [H,W]tensor trainId
        if label is not None:
            label = torch.from_numpy(np.asarray(label))

        return image, label

    pass

class TrainTransform:
    def __init__(self):
        self.crop = PairCrop(offsets=(690, None))
        self.resize = PairResize(size=256)
        self.hflip = PairRandomHFlip()
        self.color_adjust = AdjustColor()
        self.normalize = NormalizeToTensor()
        self.cutout = RandomCutout()

    def __call__(self, image, label):
        image, label = self.crop(image, label)
        image, label = self.resize(image, label)
        image, label = self.hflip(image, label)
        image = self.color_adjust(image)
        image, label = self.normalize(image, label)
        image = self.cutout(image)
        return image, label

class TrainTransform:
    def __init__(self, size=256):
        self.pair_resize = PairResize(size)
        self.image_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, image, label):
        image, label = self.pair_resize(image, label)
        image = self.image_transform(image)
        label = torch.from_numpy(np.array(label)).long()
        return image, label

# class TestTransform(TrainTransform):
#     # For this example, we're using the same transform for test as for train
#     pass
class TestTransform:
    def __init__(self):
        self.crop = PairCrop(offsets=(690, None))
        self.resize = PairResize(size=256)
        self.normalize = NormalizeToTensor()

    def __call__(self, image, label):
        image, label = self.crop(image, label)
        image, label = self.resize(image, label)
        image, label = self.normalize(image, label)
        return image, label
