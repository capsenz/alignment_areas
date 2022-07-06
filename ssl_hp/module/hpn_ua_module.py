from ctypes import alignment
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
from torch.optim.lr_scheduler import CosineAnnealingLR



def align_loss(x, y, alpha=2):
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()

def uniform_loss(x, t=2):
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()

def interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, bt] + s[1:]), 1, 0), [-1] + s[1:])


def de_interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([bt, -1] + s[1:]), 1, 0), [-1] + s[1:])

class SemiUA(ClassifierModule):
    def __init__(self, hparams, classifier, loaders, f_loss, polars):
        super(SemiUA, self).__init__(hparams, classifier, loaders, f_loss, polars)

        self.stage = 2
        
        self.polars= self.polars.to(self.device)

    def training_step(self, batch, batch_nb):

        if self.stage == 1: 
            
            x_label, y = batch[0]
            x_light, x_heavy, y_ul = batch[1]

            # inputs = interleave(
            #     torch.cat((x_label, x_light, x_heavy)), 3)

            y_hat = self.classifier(x_label)

            # y_hat = de_interleave(y_hats, 3)
            # y_hat = y_hats[:self.hparams.batch_size]
            target = self.polars[y]

            loss = (1 - self.f_loss(y_hat, target.to(self.device))).pow(2).mean()
            num = len(y)

            # calculate accuracy
            output = y_hat.float()
            output = self.classifier.predict(output).float()
            pred = output.max(1, keepdim=True)[1]
            acc = pred.eq(y.view_as(pred)).sum().item()
            self.log('train_acc', acc / float(num), on_step=True, on_epoch=True)
            self.log('sup_loss', loss)

            return {"loss": loss, "train_acc": acc / float(num), "train_num": num}
        

        if self.stage == 2: 
        
            x_label, y = batch[0]
            x_light, x_heavy, y_ul = batch[1]

            bt = x_label.size(0)
            mu = int(x_light.size(0) // bt)

            imgs = torch.cat([x_label, x_light, x_heavy], dim=0).cuda()
            imgs = interleave(imgs, 2 * mu + 1)
            logits = self.classifier(imgs)
            logits = de_interleave(logits, 2 * mu + 1)

            y_hat = logits[:self.hparams.batch_size]
            y_hat_light, y_hat_heavy = logits[self.hparams.batch_size:].chunk(2)
            del logits

            target = self.polars[y]

            # labeled loss
            sup_loss = (1 - self.f_loss(y_hat, target.to(self.device))).pow(2).mean()
            self.log('sup_loss', sup_loss)

            align_loss_train = align_loss(y_hat_light, y_hat_heavy, alpha=2)
            unif_loss= (uniform_loss(y_hat_light, t=2) + uniform_loss(y_hat_heavy, t=2)) / 2

            # check thresholds         
            self.log("ua_loss", align_loss_train + unif_loss)

            # combined loss
            comb_loss = sup_loss + align_loss_train + unif_loss
            self.log("loss", comb_loss)

            num = len(y)

            # calculate accuracy
            output = y_hat.float()
            output = self.classifier.predict(output).float()
            pred = output.max(1, keepdim=True)[1]

            acc = pred.eq(y.view_as(pred)).sum().item()
            self.log('train_acc', acc / float(num), on_step=True, on_epoch=True)

            return {"loss": comb_loss, "train_acc": acc/float(num), "train_num": num}

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
    
    # def on_train_epoch_start(self):
    #     step_size = 1 / self.trainer.max_epochs
    #     if self.current_epoch == 0:
    #         self.loss_weight = step_size
    #     else:
    #         self.loss_weight += step_size
    #     self.log("ua_loss_weight", self.loss_weight)

        #stage scheduling 
        # if self.current_epoch >= 35: #/ self.trainer.max_epochs
        #     self.stage = 2
        # self.log("stage", self.stage)

    def on_train_epoch_start(self):
        print()

    def configure_optimizers(self):
        # REQUIRED
        # this is ugly but i do not know any other way
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        
        scheduler = CosineAnnealingLR(opt, T_max=self.trainer.estimated_stepping_batches)

        return [opt] , [scheduler]