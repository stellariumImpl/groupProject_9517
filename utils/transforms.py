import random
import torch
import numpy as np
from torchvision import transforms
from torchvision.transforms import functional as TF
import cv2
from PIL import Image

class TrainTransform:
    def __init__(self, size=256, gaussian_prob=0.5, gaussian_kernel=(5, 5), gaussian_sigma=(0.1, 2.0)):
        self.size = (size, size)  # Changed to tuple
        self.gaussian_prob = gaussian_prob
        self.gaussian_kernel = gaussian_kernel
        self.gaussian_sigma = gaussian_sigma

    def __call__(self, image, label):
        # Resize
        image = TF.resize(image, self.size)
        label = TF.resize(label, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # Random horizontal flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            label = TF.hflip(label)

        # Random rotation
        angle = random.uniform(-10, 10)
        image = TF.rotate(image, angle)
        label = TF.rotate(label, angle, interpolation=TF.InterpolationMode.NEAREST)

        # Random Gaussian blur
        if random.random() < self.gaussian_prob:
            sigma = random.uniform(self.gaussian_sigma[0], self.gaussian_sigma[1])
            image_np = np.array(image)
            image_np = cv2.GaussianBlur(image_np, self.gaussian_kernel, sigma)
            image = Image.fromarray(image_np)

        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label = torch.from_numpy(np.array(label)).long()

        # Verify shapes
        assert image.shape[0] == 3, f"Image should have 3 channels, got {image.shape[0]}"
        assert image.shape[1] == image.shape[2] == self.size[0], f"Image should be square with size {self.size[0]}, got shape {image.shape}"
        assert label.shape == image.shape[1:], f"Label shape {label.shape} doesn't match image shape {image.shape[1:]}"

        return image, label

class TestTransform:
    def __init__(self, size=256):
        self.size = (size, size)  # Changed to tuple

    def __call__(self, image, label):
        # Resize
        image = TF.resize(image, self.size)
        label = TF.resize(label, self.size, interpolation=TF.InterpolationMode.NEAREST)

        # Convert to tensor and normalize
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        label = torch.from_numpy(np.array(label)).long()

        # Verify shapes
        assert image.shape[0] == 3, f"Image should have 3 channels, got {image.shape[0]}"
        assert image.shape[1] == image.shape[2] == self.size[0], f"Image should be square with size {self.size[0]}, got shape {image.shape}"
        assert label.shape == image.shape[1:], f"Label shape {label.shape} doesn't match image shape {image.shape[1:]}"

        return image, label

class AdvancedAugmentation:
    def __init__(self, size=224, p=0.5):
        self.size = (size, size)
        self.p = p

    def __call__(self, image, mask):
         # Make sure the input is a PIL image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(np.uint8(image))
        if not isinstance(mask, Image.Image):
            mask = Image.fromarray(np.uint8(mask))

        # adjust size
        image = TF.resize(image, self.size, interpolation=TF.InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.size, interpolation=TF.InterpolationMode.NEAREST)

        if random.random() < self.p:
            # random flipping
            if random.random() < 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # random vertical flip
            if random.random() < 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
            
            # random rotation
            angle = random.uniform(-10, 10)
            image = TF.rotate(image, angle, interpolation=TF.InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=TF.InterpolationMode.NEAREST)
            
            # random colour dithering
            if random.random() < 0.5:
                image = TF.adjust_brightness(image, brightness_factor=random.uniform(0.8, 1.2))
                image = TF.adjust_contrast(image, contrast_factor=random.uniform(0.8, 1.2))
                image = TF.adjust_saturation(image, saturation_factor=random.uniform(0.8, 1.2))
                image = TF.adjust_hue(image, hue_factor=random.uniform(-0.1, 0.1))

            # Random Gaussian Blur
            if random.random() < 0.25:
                image = TF.gaussian_blur(image, kernel_size=3)

        # convert to tensor and normalise
        image = TF.to_tensor(image)
        image = TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        mask = torch.from_numpy(np.array(mask)).long()

        # Validate shapes
        assert image.shape[0] == 3, f"Image should have 3 channels, but got {image.shape[0]}"
        assert image.shape[1] == image.shape[2] == self.size[0], f"Image should be a square of size {self.size[0]}, but got shape {image.shape}"
        assert mask.shape == image.shape[1:], f"Mask shape {mask.shape} doesn't match image shape {image.shape[1:]}"

        return image, mask
