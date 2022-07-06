import torch
from torch.utils.data import Subset, DataLoader
import pytorch_lightning as pl
from torchvision.datasets import STL10
import numpy as np
import torchvision.transforms as T
from pytorch_lightning.trainer.supporters import CombinedLoader
import os 
from torchvision.utils import make_grid
from torchvision.io import read_image
from pathlib import Path
import torchvision.transforms.functional as F
from typing import Any, Callable, Optional, Tuple, cast
from PIL import Image
from torchvision.datasets.utils import check_integrity, download_and_extract_archive, verify_str_arg
from torchvision.datasets import VisionDataset

class STL10_cust(VisionDataset):
    """`STL10 <https://cs.stanford.edu/~acoates/stl10/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where directory
            ``stl10_binary`` exists.
        split (string): One of {'train', 'test', 'unlabeled', 'train+unlabeled'}.
            Accordingly dataset is selected.
        folds (int, optional): One of {0-9} or None.
            For training, loads one of the 10 pre-defined folds of 1k samples for the
            standard evaluation procedure. If no value is passed, loads the 5k samples.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """

    base_folder = "stl10_binary"
    url = "http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz"
    filename = "stl10_binary.tar.gz"
    tgz_md5 = "91f7769df0f17e558f3565bffb0c7dfb"
    class_names_file = "class_names.txt"
    folds_list_file = "fold_indices.txt"
    train_list = [
        ["train_X.bin", "918c2871b30a85fa023e0c44e0bee87f"],
        ["train_y.bin", "5a34089d4802c674881badbb80307741"],
        ["unlabeled_X.bin", "5242ba1fed5e4be9e1e742405eb56ca4"],
    ]

    test_list = [["test_X.bin", "7f263ba9f9e0b06b93213547f721ac82"], ["test_y.bin", "36f9794fa4beb8a2c72628de14fa638e"]]
    splits = ("train", "train+unlabeled", "unlabeled", "test")

    def __init__(
        self,
        root: str,
        split: str = "train",
        folds: Optional[int] = None,
        transform: Optional[Any] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, transform=transform, target_transform=target_transform)
        self.split = verify_str_arg(split, "split", self.splits)
        self.folds = self._verify_folds(folds)

        if download:
            self.download()
        elif not self._check_integrity():
            raise RuntimeError("Dataset not found or corrupted. You can use download=True to download it")

        # now load the picked numpy arrays
        self.labels: Optional[np.ndarray]
        if self.split == "train":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)

        elif self.split == "train+unlabeled":
            self.data, self.labels = self.__loadfile(self.train_list[0][0], self.train_list[1][0])
            self.labels = cast(np.ndarray, self.labels)
            self.__load_folds(folds)
            unlabeled_data, _ = self.__loadfile(self.train_list[2][0])
            self.data = np.concatenate((self.data, unlabeled_data))
            self.labels = np.concatenate((self.labels, np.asarray([-1] * unlabeled_data.shape[0])))

        elif self.split == "unlabeled":
            self.data, _ = self.__loadfile(self.train_list[2][0])
            self.labels = np.asarray([-1] * self.data.shape[0])
        else:  # self.split == 'test':
            self.data, self.labels = self.__loadfile(self.test_list[0][0], self.test_list[1][0])

        class_file = os.path.join(self.root, self.base_folder, self.class_names_file)
        if os.path.isfile(class_file):
            with open(class_file) as f:
                self.classes = f.read().splitlines()


    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        target: Optional[int]
        if self.labels is not None:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], None

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.labels is not None:
            img = self.transform(img)
            return img, target
        
        if self.labels is None:
            # heavy transforms for fixmatch
            # all the conversions are awkward but i do not know any other way

            img_w = self.transform[0](img)
            img_h = self.transform[1](img)

            return img_w, img_h

    def _verify_folds(self, folds: Optional[int]) -> Optional[int]:
        if folds is None:
            return folds
        elif isinstance(folds, int):
            if folds in range(10):
                return folds
            msg = "Value for argument folds should be in the range [0, 10), but got {}."
            raise ValueError(msg.format(folds))
        else:
            msg = "Expected type None or int for argument folds, but got type {}."
            raise ValueError(msg.format(type(folds)))

    def __len__(self) -> int:
        return self.data.shape[0]

    def __loadfile(self, data_file: str, labels_file: Optional[str] = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        labels = None
        if labels_file:
            path_to_labels = os.path.join(self.root, self.base_folder, labels_file)
            with open(path_to_labels, "rb") as f:
                labels = np.fromfile(f, dtype=np.uint8) - 1  # 0-based

        path_to_data = os.path.join(self.root, self.base_folder, data_file)
        with open(path_to_data, "rb") as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96))
            images = np.transpose(images, (0, 1, 3, 2))

        return images, labels

    def _check_integrity(self) -> bool:
        root = self.root
        for fentry in self.train_list + self.test_list:
            filename, md5 = fentry[0], fentry[1]
            fpath = os.path.join(root, self.base_folder, filename)
            if not check_integrity(fpath, md5):
                return False
        return True

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return
        download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)
        self._check_integrity()

    def extra_repr(self) -> str:
        return "Split: {split}".format(**self.__dict__)

    def __load_folds(self, folds: Optional[int]) -> None:
        # loads one of the folds if specified
        if folds is None:
            return
        path_to_folds = os.path.join(self.root, self.base_folder, self.folds_list_file)
        with open(path_to_folds) as f:
            str_idx = f.read().splitlines()[folds]
            list_idx = np.fromstring(str_idx, dtype=np.int64, sep=" ")
            self.data = self.data[list_idx, :, :, :]
            if self.labels is not None:
                self.labels = self.labels[list_idx]


        


# use this to load data as in self-supervised paper+
class TwoAugUnlabDataset(torch.utils.data.Dataset):
    """Returns two augmentation and no labels."""
    def __init__(self, dataset, transform_weak, transform_heavy):
        self.dataset = dataset
        # TODO: see if that is viable or if we could do that on batch somehow
        self.transform_weak = transform_weak
        self.transform_heavy = transform_heavy

    def __getitem__(self, index):
        image, y = self.dataset[index]
        return self.transform_weak(image), self.transform_heavy(image)

    def __len__(self):
        return len(self.dataset)


class LabDataset(torch.utils.data.Dataset):
    """Returns two augmentation and no labels."""
    def __init__(self, dataset, transform):
        self.dataset = dataset
        # TODO: see if that is viable or if we could do that on batch somehow
        # self.transform = transform

    def __getitem__(self, index):
        image, y = self.dataset[index]
        return image, y

    def __len__(self):
        return len(self.dataset)


STL_MEAN = (0.44087801806139126, 0.42790631331699347, 0.3867879370752931)        
STL_STD = (0.26826768628079806, 0.2610450402318512, 0.26866836876860795)


class STL10HPNFixmatch(pl.LightningDataModule):
    def __init__(self, data_dir, num_workers, batch_size):

        super().__init__()
        self.prepare_data_per_node = False

        self.n_classes = 10
        
        self.data_dir = data_dir
        self.num_workers = num_workers
        self.batch_size = batch_size


        # unlabeled transforms
        self.weak_transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=96, padding=int(96*0.125), padding_mode='reflect'),
        T.Normalize(STL_MEAN, STL_STD)
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
        T.Normalize(STL_MEAN, STL_STD)
        ])

        self.train_transform = T.Compose([
        T.ToTensor(),
        T.RandomHorizontalFlip(),
        T.RandomCrop(size=96, padding=int(96*0.125), padding_mode='reflect'),
        T.Normalize(STL_MEAN, STL_STD),
        ])

        self.test_transform = T.Compose([
        T.Normalize(STL_MEAN, STL_STD),
        ])
    def prepare_data(self):

        # load datasets
        # TODO: Maybe change transforms
        print("train")
        self.train_set_lab = STL10_cust(
            self.data_dir, split="train", download=True, transform=self.weak_transform
        )

        print("ulab")
        self.train_set_ulab = STL10_cust(
            self.data_dir, split="unlabeled", download=True, transform=[self.weak_transform, self.heavy_transform]
        )

        print("test")
        # TODO: Maybe change transforms
        self.test_set = STL10(
            self.data_dir, split="test", download=True, transform=self.test_transform
        )
        
        # # define subsets then load as two dataloaders
        # self.label_train = Subset(self.train_set, self.labeled_indices)
        # self.unlabel_train = Subset(self.train_set, self.unlabeled_indices)



    def setup(self, stage):
        print("lol")
        # TODO: Maybe change transforms
        # self.label_train = LabDataset(self.train_set_lab, self.weak_transform)
        # # TODO: Maybe change transforms
        # self.unlabel_train = TwoAugUnlabDataset(self.train_set_ulab, self.weak_transform, self.heavy_transform)
        # TODO: Need to do transforms here!!!
        # self.val_set = Subset(self.train_set, self.val_indices)

    def train_dataloader(self):
        # data loaders
        loader_labeled = DataLoader(self.train_set_lab,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,)

        loader_unlabeled = DataLoader(self.train_set_ulab,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,)
        
        # merge loaders
        loaders = [loader_labeled, loader_unlabeled]

        # combined_loader = CombinedLoader(loaders, mode="max_size_cycle")
        return loaders


    def val_dataloader(self):

        loader_test = DataLoader(self.test_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,)

        return loader_test

    def test_dataloader(self):
        loader_test = DataLoader(self.test_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=True,)
        return loader_test

    # @property
    # def n_classes(self):
    #     # self._n_class should be defined in _prepare_train_dataset()
    #     return 10

    # @property
    # def num_labeled_data(self):
    #     assert self.train_set is not None, (
    #         "Load train data before calling %s" % self.num_labeled_data.__name__
    #     )
    #     return len(self.labeled_indices)

    # @property
    # def num_unlabeled_data(self):
    #     assert self.train_set is not None, (
    #         "Load train data before calling %s" % self.num_unlabeled_data.__name__
    #     )
    #     return len(self.unlabeled_indices)

    # @property
    # def num_val_data(self):
    #     assert self.train_set is not None, (
    #         "Load train data before calling %s" % self.num_val_data.__name__
    #     )
    #     return len(self.val_indices)

    # @property
    # def num_test_data(self):
    #     assert self.test_set is not None, (
    #         "Load test data before calling %s" % self.num_test_data.__name__
    #     )
    #     return len(self.test_set)
