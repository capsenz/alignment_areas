import numpy as np
from .base_data import (
    SemiDataModule,
    SupervisedDataModule,
)

import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import GTSRB

MEAN = (0.4914, 0.4822, 0.4465)
STD =  (0.2023, 0.1994, 0.2010)

class SemiGTSRBModule(SemiDataModule):
    def __init__(
        self,
        args,
        data_root,
        num_workers,
        batch_size,
        num_labeled,
        num_val,
        n_classes
    ):
        n_classes = 43
        super(SemiGTSRBModule, self).__init__(
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
        T.Resize((32, 32)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        T.Normalize(MEAN, STD), 
        ])

        # heavy transforms for fixmatch
        # all the conversions are awkward but i do not know any other way
        self.heavy_transform = T.Compose([
        T.ToTensor(),
        T.Resize((32, 32)),
        T.ToPILImage(),
        T.PILToTensor(),
        T.RandAugment(),
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(MEAN, STD), 
        ])

        self.train_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.Resize((32, 32)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
        T.Normalize(MEAN, STD),
        ])

        self.test_transform = tv.transforms.Compose([
        T.Resize((32, 32)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
        ])

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()

        self.train_set = GTSRB(
            self.data_root, split="train", download=True, transform=None
        )

        self.test_set = GTSRB(
            self.data_root, split="test", download=True, transform=self.test_transform
        )


class SupervisedGTSRBModule(SupervisedDataModule):
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
        n_classes = 43
        super(SupervisedGTSRBModule, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )

        self.train_transform = tv.transforms.Compose([
                T.Resize((32, 32)),
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=32, padding=int(32*0.125), padding_mode='reflect'),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(MEAN, STD),
            ]
        )

        self.test_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                T.Resize((32, 32)),
                tv.transforms.Normalize(MEAN, STD),
            ]
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()
        self.train_set = GTSRB(
            self.data_root, split="train", download=True, transform=None
        )

        self.test_set = GTSRB(
            self.data_root, split="test", download=True, transform=self.test_transform
        )