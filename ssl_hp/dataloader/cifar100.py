import numpy as np
from .base_data import (
    SemiDataModule,
    SupervisedDataModule,
)

import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import CIFAR100
import torch


CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2471, 0.2435, 0.2616)


class SemiCIFAR100Module(SemiDataModule):
    def __init__(
        self,
        args,
        data_root,
        num_workers,
        batch_size,
        num_labeled,
        num_val,
        n_classes,
    ):
        n_classes = 100
        super(SemiCIFAR100Module, self).__init__(
            data_root,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            n_classes,
        )


        # unlabeled transforms
        self.weak_transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

        # heavy transforms for fixmatch
        # all the conversions are awkward but i do not know any other way
        self.heavy_transform = T.Compose([
        T.ToTensor(),
        T.ToPILImage(),
        T.PILToTensor(),
        T.RandAugment(magnitude=9), # int(torch.randint(1,20,(1,)))
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD)
        ])

        self.train_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        self.test_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()

        self.train_set = CIFAR100(
            self.data_root, train=True, download=True, transform=None
        )

        self.test_set = CIFAR100(
            self.data_root, train=False, download=True, transform=self.test_transform
        )


class SupervisedCIFAR100Module(SupervisedDataModule):
    def __init__(
        self,
        args,
        data_root,
        num_workers,
        batch_size,
        num_labeled,
        num_val,
        num_augments,
    ):
        n_classes = 100
        super(SupervisedCIFAR100Module, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )

        self.train_transform = tv.transforms.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

        self.test_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
            ]
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()
        self.train_set = CIFAR100(
            self.data_root, train=True, download=True, transform=None
        )

        self.test_set = CIFAR100(
            self.data_root, train=False, download=True, transform=self.test_transform
        )