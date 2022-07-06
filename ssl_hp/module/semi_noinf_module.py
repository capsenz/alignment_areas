import os
import torch
import torchmetrics
import collections
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from ssl_hp.module.classifier_module import ClassifierModule
from ssl_hp.utils.torch_utils import (
    customized_weight_decay,
    WeightDecayModule,
    split_weight_decay_weights,
)

# Use the style similar to pytorch_lightning (pl)
# Codes will revised to be compatible with pl when pl has all the necessary features.

# Codes borrowed from
# https://colab.research.google.com/drive/1F_RNcHzTfFuQf-LeKvSlud6x7jXYkG31#scrollTo=x-34xKCI40yW


class SemiNonInfClassifierModule(ClassifierModule):
    def __init__(self, hparams, classifier, loaders, f_loss, polars):
        super(SemiNonInfClassifierModule, self).__init__(hparams, classifier, loaders, f_loss, polars)
        # this is new, check if it works correctly 

    def training_step(self, batch, batch_nb):
        # REQUIRED
        labeled_batch, unlabeled_batch = batch[0], batch[1]
        labeled_x, labeled_y = labeled_batch
        
        target = self.polars[labeled_y]

        y_hat = self.classifier(labeled_x)
        loss = (1 - self.f_loss(y_hat, target.to(self.device))).pow(2).sum()
        num = len(labeled_y)

        # calculate accuracy
        output = y_hat.float()
        output = self.classifier.predict(output).float()
        pred = output.max(1, keepdim=True)[1]
        acc = pred.eq(labeled_y.view_as(pred)).sum().item()
        self.log('train_acc', acc / float(num), on_step=True, on_epoch=True)
        self.log('loss', loss)

        return {"loss": loss, "train_acc": acc / float(num), "train_num": num}