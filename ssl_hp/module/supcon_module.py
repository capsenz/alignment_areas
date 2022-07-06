from ctypes import alignment
import os

from zmq import device
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from .losses import SupConLoss, SupConSemiLoss


class SemiSupCon(pl.LightningModule):
    
    def __init__(self, hparams, model, classifier, loaders):
        super(SemiSupCon, self).__init__()
        
        self.hparams.update(vars(hparams))
        self.model = model
        self.classifier = classifier
        self.loaders = loaders

        self.stage = 2
        self.threshold = self.hparams.threshold
        
        self.sup_criterion = SupConLoss()
        self.semi_criterion = SupConSemiLoss()

        self.val_criterion = torch.nn.CrossEntropyLoss()

    def training_step(self, batch, batch_nb):
        
        # training stage sup 
        if self.stage == 1: 
            
            x_label_a, x_label_b, y = batch[0]
            # x_light, x_heavy, y_ul = batch[1]

            images = torch.cat([x_label_a, x_label_b], dim=0)

            if torch.cuda.is_available():
                images = images.cuda(non_blocking=True)
                y = y.cuda(non_blocking=True)
            
            bsz = y.shape[0]

            features = self.model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = self.sup_criterion(features, y)

            # # calculate accuracy
            # output = y_hat.float()
            # output = self.classifier.predict(output).float()
            # pred = output.max(1, keepdim=True)[1]
            # acc = pred.eq(y.view_as(pred)).sum().item()
            # self.log('train_acc', acc / float(num), on_step=True, on_epoch=True)
            # self.log('sup_loss', loss)

            return {"loss": loss} # , "train_acc": acc / float(num), "train_num": num

        # trianing stage semi
        if self.stage == 2:

            x_label_a, x_label_b, y = batch[0]
            x_light, x_heavy, y_ul = batch[1]

            images_lab = torch.cat([x_label_a, x_label_b], dim=0)
            images_ulab = torch.cat([x_light, x_heavy], dim=0)

            bsz = y.shape[0]

            # supervised loss
            features = self.model(images_lab)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss_sup = self.sup_criterion(features, y)
            self.log('sup_loss', loss_sup)

            # unsupervised loss
            features_ulab = self.model(images_ulab)
            f1_ulab, f2_ulab = torch.split(features_ulab, [bsz, bsz], dim=0)
            features_ulab = torch.cat([f1_ulab.unsqueeze(1), f2_ulab.unsqueeze(1)], dim=1)

            # pseudo_label = self.classifier(f1_ulab.detach())
            # # probably check how pseudolabel looks like 
            # max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            # mask = max_probs.ge(self.hparams.threshold).float().to(self.device)
            # self.log("mask_sum", mask.sum())

            # using collapsing sup criterion here
            loss_semi = self.sup_criterion(features_ulab)
            self.log('semi_loss', loss_semi)

            loss_comb = loss_sup + loss_semi
            self.log('loss', loss_comb)

            return {"loss": loss_comb}

        # classifier stage
        if self.stage == 3:
            x_label_a, x_label_b, y = batch[0]
            x_light, x_heavy, y_ul = batch[1]

            output = self.classifier(self.model.encoder(x))
            pred = output.max(1, keepdim=True)[1]
            val_acc = pred.eq(y.view_as(pred)).sum().item()
            num = len(y)
            loss = self.val_criterion(output, y)
            self.log('train_class_acc', val_acc / float(num), on_step=True, on_epoch=True)
            self.log('train_loss_ce', loss, on_step=True, on_epoch=True)
            
            return {"loss": loss, "train_class_acc": val_acc, "val_num": num}

    def validation_step(self, batch, *args):
        x, y = batch 
        output = self.classifier(self.model.encoder(x))
        pred = output.max(1, keepdim=True)[1]
        val_acc = pred.eq(y.view_as(pred)).sum().item()
        num = len(y)
        loss = self.val_criterion(output, y)
        self.log('valid_acc', val_acc / float(num), on_step=True, on_epoch=True)
        self.log('valid_loss_ce', loss, on_step=True, on_epoch=True)
        
        return {"val_loss": loss, "val_acc": val_acc, "val_num": num}


    def test_step(self, batch, *args):
        x, y = batch 
        output = self.classifier(self.model.encoder(x))
        test_acc = output.eq(y.view_as(output)).sum().item()
        num = len(y)
        loss = self.val_criterion(output, y)
        self.log('test_acc', test_acc / float(num), on_step=True, on_epoch=True)
        self.log('test_loss_ce', loss, on_step=True, on_epoch=True)
        
        return {"test_loss": loss, "test_acc": test_acc, "val_num": num}

    def on_train_epoch_start(self):

        
        # stage scheduling 
        if (self.current_epoch + 1) / self.trainer.max_epochs >= 0.66:
            self.stage = 3
        self.log("stage", self.stage)



    def configure_optimizers(self):
        # REQUIRED
        # this is ugly but i do not know any other way
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        scheduler = {
            'scheduler': ReduceLROnPlateau(opt, patience=20, mode='min', threshold_mode='rel', min_lr=8e-5),
            'monitor': 'loss',
            "interval": "epoch",
            'name':'lr'
        }
        return [opt], [scheduler]

        