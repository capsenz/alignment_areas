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
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR


class ClassifierModule(pl.LightningModule):
    def __init__(self, hparams, classifier, loaders, f_loss, polars):
        super(ClassifierModule, self).__init__()
        # this is new, check if it works correctly 
        # more elegant solution might be: https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html#lightningmodule-hyperparameters
        self.hparams.update(vars(hparams))
        self.classifier = classifier
        self.f_loss = f_loss
        self.loaders = loaders
        self.polars = polars.to(self.device)
        self.best_dict = {
            "val_acc": 0,
        }

    # def accuracy(self, y_hat, y):
    #     return 100 * (torch.argmax(y_hat, dim=-1) == y).float().mean()

    def training_step(self, batch, batch_nb):
        # REQUIRED
        x, y = batch
        target = self.polars[y]

        y_hat = self.classifier(x)
        loss = (1 - self.f_loss(y_hat, target.to(self.device))).pow(2).sum()
        num = len(y)

        # calculate accuracy
        output = y_hat.float()
        output = self.classifier.predict(output).float()
        pred = output.max(1, keepdim=True)[1]
        acc = pred.eq(y.view_as(pred)).sum().item()
        self.log('train_acc', acc / float(num), on_step=True, on_epoch=True)
        self.log('loss', loss)

        return {"loss": loss, "train_acc": acc / float(num), "train_num": num}
    

    def validation_step(self, batch, *args):
        # OPTIONAL
        x, y = batch
        y_hat = self.classifier(x)

        # calculate accuracy
        output = y_hat.float()
        output = self.classifier.predict(output).float()
        pred = output.max(1, keepdim=True)[1]
        val_acc = pred.eq(y.view_as(pred)).sum().item()
        num = len(y)
        loss = (1 - self.f_loss(pred.float(), y.float().to(self.device))).pow(2).sum()
        self.log('valid_acc', val_acc / float(num), on_step=True, on_epoch=True)
        self.log('valid_loss', loss, on_step=True, on_epoch=True)

        return {"val_loss": loss, "val_acc": val_acc, "val_num": num}

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        x, y = batch
        y_hat = self.classifier(x)

        # calc acc
        output = y_hat.float()
        output = self.classifier.predict(output).float()
        pred = output.max(1, keepdim=True)[1] 
        test_acc = pred.eq(y.view_as(pred)).sum().item()
        num = len(y)
        self.log('test_acc', test_acc / float(num), on_step=True, on_epoch=True)
        

        return {
            "test_loss": F.cross_entropy(y_hat, y),
            "test_acc": test_acc,
            "test_num": num,
        }

    def on_train_start(self):
        print(self.trainer.estimated_stepping_batches)
    
    # def on_train_epoch_start(self):
    #     step_size = 1 / self.trainer.max_epochs
    #     if self.current_epoch == 0:
    #         self.loss_weight = step_size
    #     else:
    #         self.loss_weight += step_size
        # self.log("ua_loss_weight", self.loss_weight)

        # stage scheduling 
        # if (self.current_epoch + 1) / self.trainer.max_epochs >= 0.25:
        #     self.stage = 2
        # self.log("stage", self.stage)

    def configure_optimizers(self):
        # REQUIRED
        # this is ugly but i do not know any other way
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        
        scheduler = CosineAnnealingLR(opt, T_max=self.trainer.estimated_stepping_batches)

        return [opt] , [scheduler]