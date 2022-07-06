import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import torchvision.transforms as T
import kornia.augmentation as A

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)

class DataAugmentationLight(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self, apply_color_jitter: bool = False) -> None:
        super().__init__()
        self._apply_color_jitter = apply_color_jitter

        self.transforms = nn.Sequential(
            T.RandomHorizontalFlip(p=0.5),
            T.RandomAffine(degrees=0, translate=(0.125, 0.125)),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW light
        if self._apply_color_jitter:
            x_out = self.jitter(x_out)
        return x_out

class DataAugmentationHeavy(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        self.transforms = T.Compose([
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor()]
            # T.ColorJitter(brightness=.5, hue=.3, saturation=0.5, contrast=0.5),
            # T.RandomHorizontalFlip(p=0.4),
            # T.RandomAffine(degrees=0, translate=(0.125, 0.125)),
            # T.RandomPerspective(distortion_scale=0.5, p=0.4), 
            # T.RandomRotation(degrees=(0, 180)), 
            # T.RandomInvert(p=0.4), 
            # # T.RandomPosterize(bits=2, p=0.4),
            # # T.RandomSolarize(threshold=192.0, p=0.4),
            # T.RandomAdjustSharpness(sharpness_factor=0, p=0.4),
            # T.RandomVerticalFlip(p=0.4)
            # # A.RandomGaussianBlur((3, 3), (1.5, 1.5), p=0.4),
            # # A.RandomGaussianNoise(p=0.4),
            # # A.RandomVerticalFlip(p=0.4),
            # # A.RandomMixUp(p=0.4),
            # # T.RandAugment()
        )

    def forward(self, x): #: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW heavy
        return x_out

class DataAugmentationSup(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        self.transforms = T.Compose(
            [T.RandomCrop(32, padding=4, padding_mode="reflect"),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    def forward(self, x): #: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW sup
        return x_out



