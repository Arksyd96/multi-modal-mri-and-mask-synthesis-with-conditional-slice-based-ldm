import os
import random
from datetime import datetime
import numpy as np
from omegaconf import OmegaConf

import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger

from modules.data_preprocessing import BRATSDataModule
from modules.models.autoencoder import VAE
from modules.utils import set_seed

os.environ['WANDB_API_KEY'] = 'wandb_api_key'
DEBUG = False

if __name__ == "__main__":
    config = {}

    config['seed'] = 42
    set_seed(config['seed'], deterministic=False)
    
    if not DEBUG:
        # --------------- Settings --------------------
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        config['save_dir'] = "{}/runs/first-stage-{}".format(
            os.path.curdir, str(current_time)
        )
        os.makedirs(config['save_dir'], exist_ok=True)

        # --------------- Logger --------------------
        config['name'] = "run_name"
        logger = wandb_logger.WandbLogger(
            project="project_name",
            name=config['name'],
            save_dir=config['save_dir'],
        )


    # --------------- DataModule --------------------
    config['datamodule'] = {
        "data_dir": '../data/hecktor_preprocessed_128x128_PT.npy', # the dataset has to be preprocessed and saved as a numpy file before
        "train_ratio": 0.15,
        "batch_size": 16,
        "num_workers": 32,
        "shuffle": True,
        "include_radiomics": False # if True, the radiomics features will be included in the dataset (condition vector)
    }
    datamodule = BRATSDataModule(**config['datamodule'], dtype=torch.float32)

    
    # --------------- Model --------------------
    config['backbone'] = {
        "in_channels": 2,
        "out_channels": 2,
        "spatial_dims": 2,
        "emb_channels": 6,
        "hid_chs": [64, 128, 256, 512],
        "kernel_sizes": [3, 3, 3, 3],
        "strides": [1, 2, 2, 2],
        "temb_channels": None,
        "use_res_block": True,
        "use_attention": 'none',
        "embedding_loss_weight": 1e-6,
        "perceptual_loss_weight": 0.5,
        "sample_every_n_steps": 20,
        "optimizer_kwargs": {'lr': 5e-6}
    }
    model = VAE(**config['backbone'])

    # --------------- Trainer --------------------
    checkpointing = ModelCheckpoint(
        dirpath = config['save_dir'],
        monitor = 'val - ssim_epoch',
        every_n_epochs = 1,
        save_last = True,
        save_top_k = 1,
        mode = 'max'
    )

    config['trainer'] = {
        "strategy": 'ddp',
        "devices": 8,
        "num_nodes": 1,
        "precision": 32,
        "accelerator": 'gpu',
        "default_root_dir": config['save_dir'],
        "enable_checkpointing": True,
        "check_val_every_n_epoch": 1,
        "log_every_n_steps": 1,
        "min_epochs": 100,
        "max_epochs": 10000,
        "num_sanity_val_steps": 0,
    }

    if not DEBUG:
        # --------------- Save Config --------------------
        OmegaConf.save(config, f"{config['save_dir']}/config.yaml")
    
    # --------------- Start training --------------------
    trainer = Trainer(logger = logger, **config['trainer'], callbacks = [checkpointing])
    trainer.fit(model, datamodule=datamodule)
