import torchvision.transforms as T
import torch.nn as nn
import kornia.augmentation as A


class DataAugmentationUA(nn.Module):
    """Module to perform data augmentation using Kornia on torch tensors."""

    def __init__(self) -> None:
        super().__init__()

        self.transforms = T.Compose([
        A.RandomResizedCrop((32,32), scale=(0.08, 1)),
        A.RandomHorizontalFlip(),
        A.ColorJitter(0.4, 0.4, 0.4, 0.4),
        A.RandomGrayscale(p=0.2), 
        # T.ToTensor()
    ])

    def forward(self, x): #: Tensor) -> Tensor:
        x_out = self.transforms(x)  # BxCxHxW sup
        return x_out
