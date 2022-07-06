import numpy as np
from .base_data import (
    SemiDataModule,
    SupervisedDataModule,
)

import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, SubsetRandomSampler
from .CUB200_class import Cub2011

CUB_MEAN = (0.485, 0.456, 0.406)
CUB_STD =(0.229, 0.224, 0.225)


class SemiCUBModule(SemiDataModule):
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
        n_classes = 200
        super(SemiCUBModule, self).__init__(
            data_root,
            num_workers,
            batch_size,
            num_labeled,
            num_val,
            num_augments,
            n_classes,
        )


        # unlabeled transforms
        self.weak_transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        T.Normalize(CUB_MEAN, CUB_STD)
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
        T.Normalize(CUB_MEAN, CUB_STD)
        ])

        self.train_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        T.Normalize(CUB_MEAN, CUB_STD),
        ])

        self.test_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.Normalize(CUB_MEAN, CUB_STD),
        ])

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()

        self.train_set = Cub2011(
            self.data_root, train=True, download=True, transform=None
        )

        self.test_set = Cub2011(
            self.data_root, train=False, download=True, transform=self.test_transform
        )


class SupervisedCUBModule(SupervisedDataModule):
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
        super(SupervisedCUBModule, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )

        self.train_transform = tv.transforms.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(CUB_MEAN, CUB_STD),
            ]
        )

        self.test_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(CUB_MEAN, CUB_STD),
            ]
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()
        self.train_set = Cub2011(
            self.data_root, train=True, download=True, transform=None
        )

        self.test_set = Cub2011(
            self.data_root, train=False, download=True, transform=self.test_transform
        )