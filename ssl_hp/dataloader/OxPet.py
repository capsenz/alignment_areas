import numpy as np
from .base_data import (
    SemiDataModule,
    SupervisedDataModule,
)

import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import OxfordIIITPet

mean = (0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

class SemiOxPetModule(SemiDataModule):
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
        n_classes = 37
        super(SemiOxPetModule, self).__init__(
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
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=224, padding=int(224*0.125), padding_mode='reflect'),
        T.Normalize(mean, std)
        ])

        # heavy transforms for fixmatch
        # all the conversions are awkward but i do not know any other way
        self.heavy_transform = T.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.ToPILImage(),
        T.PILToTensor(),
        T.RandAugment(),
        T.ToPILImage(),
        T.ToTensor(),
        T.Normalize(mean, std)
        ])

        self.train_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=224, padding=int(224*0.125), padding_mode='reflect'),
        T.Normalize(mean, std),
        ])

        self.test_transform = tv.transforms.Compose([
        T.ToTensor(),
        T.Resize((224, 224)),
        T.Normalize(mean, std),
        ])

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()

        self.train_set = OxfordIIITPet(
            self.data_root, split="trainval", download=True, transform=None
        )

        self.test_set = OxfordIIITPet(
            self.data_root, split="test", download=True, transform=self.test_transform
        )


class SupervisedOxPetModule(SupervisedDataModule):
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
        super(SupervisedOxPetModule, self).__init__(
            data_root, num_workers, batch_size, num_labeled, num_val, n_classes
        )

        self.train_transform = tv.transforms.Compose([
                T.RandomHorizontalFlip(),
                T.RandomCrop(size=28, padding=int(224*0.125), padding_mode='reflect'),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std),
            ]
        )

        self.test_transform = tv.transforms.Compose([
                tv.transforms.ToTensor(),
                tv.transforms.Normalize(mean, std),
            ]
        )

    def prepare_data(self):
        # the transformation for train and validation dataset will be
        # done in _prepare_train_dataset()
        self.train_set = OxfordIIITPet(
            self.data_root, split="trainval", download=True, transform=None
        )

        self.test_set = OxfordIIITPet(
            self.data_root, split="test", download=True, transform=self.test_transform
        )
