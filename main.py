import os
import torch
import numpy as np
from ssl_hp.module import ClassifierModule, SemiNonInfClassifierModule, SemiUA, SemiHPNFixmatch
from ssl_hp.dataloader import (
    SemiCIFAR10Module,
    SupervisedCIFAR10Module,
    SemiCIFAR100Module,
    SupervisedCIFAR100Module,
    SemiFashionMNISTModule,
    SupervisedFashionMNISTModule,
    STL10HPNFixmatch, 
    SemiGTSRBModule, 
    SupervisedGTSRBModule,
    SemiOxPetModule,
    SupervisedOxPetModule)
from ssl_hp.models import ResNet, ResNetFMNIST
import pytorch_lightning as pl
from ssl_hp.utils import count_parameters
from ssl_hp.utils.argparser import parser, print_args
import torch.nn as nn
from pytorch_lightning.callbacks import LearningRateMonitor


if __name__ == "__main__":
    args = parser()

    gpus = args.gpus if torch.cuda.is_available() else None
    print(torch.cuda.is_available())
    # set the random seed
    pl.seed_everything(args.seed)

    # load polars and update the trainy labels
    classpolars = torch.from_numpy(np.load(args.prototypes)).float()
    print(classpolars.size())
    # TODO: Ask Pascal what we are updating here
    args.output_dims = int(args.prototypes.split("/")[-1].split("-")[1][:-1]) 

    print(args.dataset)
    # load the data and classifier
    if args.dataset == "cifar10":
        if args.learning_scenario == "supervised":
            loader_class = SupervisedCIFAR10Module

        elif args.learning_scenario == "semi":
            loader_class = SemiCIFAR10Module
            # loader_class =CIFAR10HPNFixmatch

        data_loader = loader_class(
            args,
            args.data_root,
            args.num_workers,
            args.batch_size,
            args.num_labeled,
            args.num_val,
            10
        )
    elif args.dataset == "cifar100":
        if args.learning_scenario == "supervised":
            loader_class = SupervisedCIFAR100Module

        elif args.learning_scenario == "semi":
            loader_class = SemiCIFAR100Module
            # loader_class =CIFAR10HPNFixmatch

        data_loader = loader_class(
            args,
            args.data_root,
            args.num_workers,
            args.batch_size,
            args.num_labeled,
            args.num_val,
            100
        )
    elif args.dataset == "FMNIST":
        if args.learning_scenario == "supervised":
            loader_class = SupervisedFashionMNISTModule

        elif args.learning_scenario == "semi":
            loader_class = SemiFashionMNISTModule
            # loader_class =CIFAR10HPNFixmatch

        data_loader = loader_class(
            args,
            args.data_root,
            args.num_workers,
            args.batch_size,
            args.num_labeled,
            args.num_val,
            10
        )

    elif args.dataset == "stl10":
        if args.learning_scenario == "supervised":
            loader_class = SupervisedCIFAR100Module

        elif args.learning_scenario == "semi":
            loader_class = STL10HPNFixmatch
            # loader_class =CIFAR10HPNFixmatch

        data_loader = loader_class(
            "./data",
            10, 
            64
        )

    elif args.dataset == "GTSRB":
        if args.learning_scenario == "supervised":
            loader_class = SupervisedGTSRBModule

        elif args.learning_scenario == "semi":
            loader_class = SemiGTSRBModule
            # loader_class =CIFAR10HPNFixmatch

        data_loader = loader_class(
            args,
            args.data_root,
            args.num_workers,
            args.batch_size,
            args.num_labeled,
            args.num_val,
            43
        )
    elif args.dataset == "OxPet":
        if args.learning_scenario == "supervised":
            loader_class = SupervisedOxPetModule

        elif args.learning_scenario == "semi":
            loader_class = SemiOxPetModule
            # loader_class =CIFAR10HPNFixmatch

        data_loader = loader_class(
            args,
            args.data_root,
            args.num_workers,
            args.batch_size,
            args.num_labeled,
            args.num_val,
            37
        )
    else:
        raise NotImplementedError

    if args.learning_scenario == "supervised":
        module = ClassifierModule
    # TODO: implement own algo 
    else:  # semi supervised learning algorithm
        if args.algo == "semi_non_inf":
            module = SemiNonInfClassifierModule
        elif args.algo == "semi_alignment":
            module = SemiUA
        elif args.algo == "hpn_fixmatch":
            module = SemiHPNFixmatch
        else:
            raise NotImplementedError

    # print(f"model paramters: {count_parameters(classifier)} M")
    print(module)
    # set the number of classes in the args
    setattr(args, "n_classes", data_loader.n_classes)

    data_loader.prepare_data()
    data_loader.setup(stage=None)

    if args.dataset == "FMNIST":
        classifier = ResNetFMNIST(32, output_dims=args.output_dims, polars=classpolars, multiplier=1)
    else:
        classifier = ResNet(32, output_dims=args.output_dims, polars=classpolars, multiplier=1)
    # classifier = ConvNet(64, 128, 256, 512, nr_outputs=64, polars=classpolars)
    model = module(hparams=args, classifier=classifier,
        loaders=None, f_loss=nn.CosineSimilarity(eps=1e-9), polars=classpolars) # 


    if args.todo == "train":
        print(
            f"labeled size: {data_loader.num_labeled_data}"
            f"unlabeled size: {data_loader.num_unlabeled_data}, "
            f"val size: {data_loader.num_val_data}"
        )

        save_folder = (
            f"{args.dataset}_{args.learning_scenario}_{args.algo}_{args.affix}"
        )

        tb_logger = pl.loggers.TensorBoardLogger(
            os.path.join(args.log_root, "lightning_logs_final"), name=save_folder
        )
        tb_logger.log_hyperparams(args)

        # set the path of checkpoint
        save_dir = getattr(tb_logger, "save_dir", None) or getattr(
            tb_logger, "_save_dir", None
        )

        ckpt_path = os.path.join(
            save_dir, tb_logger.name, f"version_{tb_logger.version}", "checkpoints"
        )

        # TODO: Check if it makes sense to include filename
        ckpt = pl.callbacks.ModelCheckpoint(dirpath=save_dir)

        setattr(args, "checkpoint_folder", ckpt_path)

        print_args(args)

        tb_logger.log_hyperparams(args)

            
        trainer = pl.trainer.Trainer(
            auto_select_gpus=True,
            logger=tb_logger,
            max_epochs=args.max_epochs,
            enable_checkpointing=True,
            gpus=-1,
            benchmark=True,
            profiler="simple",
            progress_bar_refresh_rate=args.progress_bar_refresh_rate,
            log_every_n_steps=15,
            reload_dataloaders_every_n_epochs=1,
            # resume_from_checkpoint = "lightning_logs_debug/cifar100_semi_hpn_fixmatch_exp_10k_uweight/version_0/checkpoints/epoch=67-step=37128.ckpt",
            callbacks=[LearningRateMonitor(logging_interval="step", log_momentum=True)], #num_sanity_val_steps=2
        )

        trainer.fit(model, datamodule=data_loader)
        trainer.test(datamodule=data_loader)
    else:
        trainer = pl.trainer.Trainer(resume_from_checkpoint=args.load_checkpoint)
        trainer.test(model, datamodule=data_loader)