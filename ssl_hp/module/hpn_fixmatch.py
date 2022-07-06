from ctypes import alignment
import os

from zmq import device
import torch
import torchvision as T
import collections
import numpy as np
from copy import deepcopy
from torch.nn import functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import DataLoader
from torchvision import transforms
import pytorch_lightning as pl
from ssl_hp.module.classifier_module import ClassifierModule
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from .losses import align_loss, uniform_loss, align_loss_semi
import numpy as np
import matplotlib.pyplot as plt

# def arrange_fig(imgs, labels, mapper):
#     fig, axs = plt.subplots(ncols=6, squeeze=False)
#     labels_mapped = np.vectorize(mapper.get)(labels.cpu().detach().numpy())
#     for i, img in enumerate(imgs):
#         img = img.detach()
#         img = TF.to_pil_image(img)
#         axs[0, i].imshow(np.asarray(img)) # [0, i]
#         axs[0, i].set_title(labels_mapped[i], fontsize='small', loc='left')
#         axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

#     return fig 

def interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([-1, bt] + s[1:]), 1, 0), [-1] + s[1:])


def de_interleave(x, bt):
    s = list(x.shape)
    return torch.reshape(torch.transpose(x.reshape([bt, -1] + s[1:]), 1, 0), [-1] + s[1:])


class SemiHPNFixmatch(ClassifierModule):
    
    def __init__(self, hparams, classifier, loaders, f_loss, polars):
        super(SemiHPNFixmatch, self).__init__(hparams, classifier, loaders, f_loss, polars)

        self.stage = 2
        self.threshold = self.hparams.threshold
        print(self.threshold)
        
        self.polars= self.polars.to(self.device)

        self.class_map = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
        

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

            loss = (1 - self.f_loss(y_hat, target.to(self.device))).pow(2).sum()
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
            x_light, x_heavy, y_ul  = batch[1] # 

            bt = x_label.size(0)
            mu = int(x_light.size(0) // bt)

            imgs = torch.cat([x_label, x_light, x_heavy], dim=0).cuda()
            imgs = interleave(imgs, 2 * mu + 1)
            logits = self.classifier(imgs)
            logits = de_interleave(logits, 2*mu+1)

            # y_hat = logits[:bt]
            # y_hat_light, y_hat_heavy = torch.split(logits[bt:], bt * self.hparams.mu)
            y_hat = logits[:self.hparams.batch_size]
            y_hat_light, y_hat_heavy = logits[self.hparams.batch_size:].chunk(2)
            del logits

            target = self.polars[y]

            # labeled loss
            sup_loss = (1 - self.f_loss(y_hat, target.to(self.device))).pow(2).sum()
            # sup_loss = align_loss(y_hat, target.to(self.device))
            self.log('sup_loss', sup_loss)

            # check thresholds
            # with torch.no_grad():
            pseudo_label = torch.mm(y_hat_light, self.polars.t().cuda())
            max_probs, targets_u = torch.max(pseudo_label, dim=-1)
            # log class distribution
            # class_counts = torch.unique(targets_u, return_counts=True)
            # for i, x in enumerate(class_counts[0]):
            #     self.log(str(i), class_counts[1][i] / len(targets_u))

            mask = max_probs.ge(self.hparams.threshold).float().to(self.device)
            
            # get targets
            pseudo_proto = self.polars[targets_u].to(self.device)

            # calc fixmatch loss 
            loss_unsup= ((1 - F.cosine_similarity(y_hat_heavy, pseudo_proto)).pow(2) * mask).sum()  

            self.log("fixmatch_loss", loss_unsup)

            # combined loss
            weight = 0.3
            comb_loss = sup_loss + weight * loss_unsup
            self.log("loss", comb_loss)
            self.log("weight", weight)

            num = len(y)

            # calculate metrics
            # align_proto_heavy = align_loss(y_hat_heavy, pseudo_proto)
            # align_ulab = align_loss(y_hat_heavy, y_hat_light)
            # uniform_light = uniform_loss(y_hat_light)
            # uniform_heavy = uniform_loss(y_hat_heavy)
            impurity = 1 - (targets_u.eq(y_ul.view_as(targets_u)) * mask).sum() / mask.sum()
            # self.log("align_ulab", align_ulab)
            # self.log("align_proto_heavy", align_proto_heavy)
            # self.log("uniform_light", uniform_light)
            # self.log("uniform_heavy", uniform_heavy)
            self.log("impurity", impurity)
            self.log("mask_mean", mask.mean())
            self.log("mask_sum", mask.sum())
            
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
        # if (self.current_epoch + 1) % 30 == 0:# (self.current_epoch + 1) % 7 == 0
        #     # get pred 
        #     # check if correct or false
        #     corrects = pred.eq(y.view_as(pred))
        #     # if false get index along dimension 1 
        #     # img_ind = [ind for ind, x in enumerate(corrects) if x == False]
        #     # then use index to query in batch 
        #     invTrans = transforms.Compose([transforms.Normalize(mean = [ 0., 0., 0. ],
        #                                              std = [ 1/0.2471, 1/0.2435, 1/0.2616 ]),
        #                         transforms.Normalize(mean = [ -0.4914, -0.4822, -0.4465 ],
        #                                              std = [ 1., 1., 1. ]),
        #                        ])
        #     inds = [ind for ind, x in enumerate(corrects.squeeze()) if x == False]
        #     pred_labels = pred.squeeze()[inds]
        #     labels = y[inds]
        #     labels_mapped = np.vectorize(self.class_map.get)(labels.cpu().detach().numpy())
        #     preds_mapped = np.vectorize(self.class_map.get)(pred_labels.cpu().detach().numpy())
        #     labels_string = ", ".join(list(labels_mapped))
        #     preds_string = ", ".join(list(preds_mapped))
        #     self.logger.experiment.add_text("MisclassifiedTrue_" + str(self.current_epoch), labels_string, global_step=self.global_step, walltime=None)
        #     self.logger.experiment.add_text("MisclassifiedPred_" + str(self.current_epoch), preds_string, global_step=self.global_step, walltime=None)
        #     imgs = x[inds]
        #     imgs = invTrans(imgs)
        #     # log_fig = arrange_fig(imgs, labels, self.class_map)
        #     self.logger.experiment.add_images("Misclassified_" + str(self.current_epoch), imgs, global_step=self.global_step, walltime=None , dataformats='NCHW') # , dataformats='NCHW'

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
    
    def on_train_epoch_start(self):
        # stage scheduling 
        if self.current_epoch >= 35:
            self.stage = 2
        self.log("stage", self.stage)

        # stage scheduling 
        # if (self.current_epoch + 1) % 10 == 0:
        #     self.hparams.threshold += 0.008
        #     self.log("threshold", self.hparams.threshold)

    def configure_optimizers(self):
        # REQUIRED
        # this is ugly but i do not know any other way
        opt = torch.optim.SGD(self.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay, momentum=self.hparams.momentum)
        
        scheduler = CosineAnnealingLR(opt, T_max=self.trainer.estimated_stepping_batches)

        return [opt] , [scheduler]