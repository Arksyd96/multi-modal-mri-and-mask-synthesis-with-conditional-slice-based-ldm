import os
from datetime import datetime
from omegaconf import OmegaConf

import torch 
import pytorch_lightning as pl
from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger

from modules.data_preprocessing import BRATSDataModule
from modules.models.autoencoder import VAE
from modules.models.unet import UNet
from modules.diffusion import DiffusionPipeline, TumorEnhancingDiffusionPipeline
from modules.scheduler import GaussianNoiseScheduler
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
        config['save_dir'] = "{}/runs/tumor-enhancing-{}".format(
            os.path.curdir, str(current_time)
        )
        os.makedirs(config['save_dir'], exist_ok=True)

        # --------------- Logger --------------------
        config['name'] = "tumor-enhancing training"
        logger = wandb_logger.WandbLogger(
            project="your_project_name",
            name=config['name'],
            save_dir=config['save_dir'],
        )

    # --------------- DataModule --------------------
    config['datamodule'] = {
        "data_dir": './data/first_stage_dataset_128x128_tumoral.npy',
        "train_ratio": 1.0,
        "batch_size": 16,
        "num_workers": 32,
        "shuffle": True,
        "include_radiomics": False
    }
    datamodule = BRATSDataModule(**config['datamodule'], dtype=torch.float32)

    # --------------- Autoencoder -----------------
    autoencoder = VAE.load_from_checkpoint('./runs/first-stage-2024_03_04_101323 [x4xhqd7x]/epoch=1505-step=78312.ckpt')

    # --------------- Denoiser --------------------
    config['denoiser'] = {
        "sample_size": (64, 64),
        "in_ch": 5,
        "out_ch": 3,
        "spatial_dims": 2,
        "hid_chs": [64, 128, 256, 512],
        "strides": [1, 2, 2, 2],
        "kernel_sizes": [3, 3, 3, 3],
        "temb_channels": 128,
        "max_period": 1000,
        "cond_embedder": None,
        "deep_supervision": False,
        "use_res_block": True,
        "use_attention": 'none'
    }
    denoiser = UNet(**config['denoiser'])

    # --------------- Diffusion Pipeline --------------------
    noise_scheduler = GaussianNoiseScheduler(
        timesteps = 1000,
        beta_start = 0.002, # 0.0001, 0.0015
        beta_end = 0.02, # 0.01, 0.0195
        schedule_strategy = 'scaled_linear'
    )

    config['diffuser'] = {
        "estimator_objective": 'x_T',
        "estimate_variance": False,
        "use_self_conditioning": False,
        "use_ema": False,
        "classifier_free_guidance_dropout": 0.0,
        "clip_x0": False,
        "sample_every_n_steps": 1500
    }

    diffuser = TumorEnhancingDiffusionPipeline( # This is a Pixel-space DDPM
        noise_estimator = denoiser, 
        noise_scheduler = noise_scheduler, 
        latent_embedder = autoencoder,
        **config['diffuser'],
        loss = torch.nn.L1Loss
    )


    # -------------- Training Initialization ---------------
    checkpointing = ModelCheckpoint(
        dirpath = config['save_dir'],
        monitor = 'train - loss_epoch',
        every_n_epochs = 50,
        save_last = True,
        save_top_k = 1,
        mode = 'min'
    )

    config['trainer'] = {
        "strategy": 'ddp',
        "devices": 8,
        "num_nodes": 1,
        "precision": 32,
        "accelerator": 'gpu',
        "default_root_dir": config['save_dir'],
        "enable_checkpointing": True,
        "log_every_n_steps": 1,
        "min_epochs": 100,
        "max_epochs": 10000,
        "num_sanity_val_steps": 0
    }

    trainer = Trainer(
        logger = logger,
        **config['trainer'],
        callbacks=[checkpointing]
    )
    
    # ---------------- Execute Training ----------------
    trainer.fit(diffuser, datamodule=datamodule)


