# transforms.py

import numpy as np
import random
import torch
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