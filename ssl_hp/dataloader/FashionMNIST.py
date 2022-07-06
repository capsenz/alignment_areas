import numpy as np
from .base_data import (
    SemiDataModule,
    SupervisedDataModule,
)

import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import FashionMNIST


CIFAR_MEAN = (0.1307,)
CIFAR_STD = (0.3081,)


class SemiFashionMNISTModule(SemiDataModule):
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
        n_classes = 10
        super(SemiFashionMNISTModule, self).__init__(
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
        T.RandomCrop(size=28, padding=int(28*0.125), padding_mode='reflect'),
        T.Normalize((0.1307,), (0.3081,))
        ])

        # heavy transforms for fixmatch
        # all the conversions are awkward but i do not know any other way
        self.heavy_transform = T.Compose([
        T.ToTensor(),
        T.ToPILImage(),
        T.PILToTensor(),
        T.RandAugment(),
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,))
        ])

        self.train_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=28, padding=int(28*0.125), padding_mode='reflect'),
        T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ])

        self.test_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.Normalize((0.1307,), (0.3081,)),
        ])

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()

        self.train_set = FashionMNIST(
            self.data_root, train=True, download=True, transform=None
        )

        self.test_set = FashionMNIST(
            self.data_root, train=False, download=True, transform=self.test_transform
        )


class SupervisedFashionMNISTModule(SupervisedDataModule):
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
        n_classes = 10
        super(SupervisedFashionMNISTModule, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )

        self.train_transform = tv.transforms.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=28, padding=int(28*0.125), padding_mode='reflect'),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.test_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()
        self.train_set = FashionMNIST(
            self.data_root, train=True, download=True, transform=None
        )

        self.test_set = FashionMNIST(
            self.data_root, train=False, download=True, transform=self.test_transform
        )