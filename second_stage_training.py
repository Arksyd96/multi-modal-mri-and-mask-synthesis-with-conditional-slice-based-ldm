import os
from datetime import datetime
import torch
from omegaconf import OmegaConf

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import wandb as wandb_logger

from modules.data_preprocessing import BRATSDataModule
from modules.models.autoencoder import ConditionMLP, VAE
from modules.models.unet import UNet
from modules.diffusion import DiffusionPipeline
from modules.scheduler import GaussianNoiseScheduler
from modules.utils import set_seed

os.environ["WANDB_API_KEY"] = "wandb_api_key"
DEBUG = False

if __name__ == "__main__":
    config = {}

    config["seed"] = 42
    set_seed(config["seed"], deterministic=False)

    if not DEBUG:
        # --------------- Settings --------------------
        current_time = datetime.now().strftime("%Y_%m_%d_%H%M%S")
        config['save_dir'] = "{}/runs/second-stage-{}".format(
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
        "data_dir": "./data/second_stage_dataset_128x128_reduced.npy",
        "train_ratio": 1.0,
        "batch_size": 8,
        "num_workers": 32,
        "shuffle": True,
        "include_radiomics": True,
    }
    datamodule = BRATSDataModule(**config['datamodule'], dtype=torch.float32)

    # --------------- Autoencoder -----------------
    # load previously train autoencoder
    latent_embedder = VAE.load_from_checkpoint('./runs/first-stage-2024_04_23_105457/epoch=1132-step=135960.ckpt')

    # --------------- Denoiser --------------------
    config['denoiser'] = {
        "sample_size": (128, 128, 64),
        "in_ch": 3,
        "out_ch": 3,
        "spatial_dims": 3,
        "hid_chs": [64, 128, 256, 512],
        "strides": [1, 2, 2, 2],
        "num_res_blocks": 2,
        "kernel_sizes": [3, 3, 3, 3],
        "temb_channels": 128,
        "max_period": 1000,
        "cond_in_features": 12,
        "cond_out_features": 128,
        "cond_hidden_dim": 512,
        "deep_supervision": False,
        "use_res_block": True,
        "use_attention": "none",  # [False, False, False, False]
    }

    denoiser = UNet(
        **config['denoiser'],
        cond_embedder=ConditionMLP(
            config['denoiser']['cond_in_features'], 
            config['denoiser']['cond_out_features'], 
            config['denoiser']['cond_hidden_dim']),
    )

    # --------------- Diffusion Pipeline --------------------
    config['scheduler'] = {
        "timesteps": config['denoiser']['max_period'],
        "beta_start": 0.002, # Possible alternative values: 0.0001, 0.0015
        "beta_end": 0.02, # Possible alternative values: 0.01, 0.0195
        "schedule_strategy": "scaled_linear"
    }
    noise_scheduler = GaussianNoiseScheduler(**config['scheduler'])

    config['diffuser'] = {
        "noise_estimator": denoiser,
        "noise_scheduler": noise_scheduler,
        "latent_embedder": None,
        "estimator_objective": "x_T",
        "estimate_variance": False,
        "use_self_conditioning": False,
        "use_ema": False,
        "classifier_free_guidance_dropout": 0.0,
        "clip_x0": False,
        "slice_based": False,
        "std_norm": None, # 0.8784011602401733 to be computed first on a batch of latents
        "sample_every_n_steps": 1000
    }

    diffuser = DiffusionPipeline(
        noise_estimator = denoiser,
        noise_scheduler = noise_scheduler,
        latent_embedder = latent_embedder,
        **config['diffuser'],
        loss = torch.nn.L1Loss
    )

    # --------------- Trainer --------------------
    checkpointing = ModelCheckpoint(
        dirpath = config['save_dir'],
        monitor = 'train - loss_epoch',
        every_n_epochs = 1,
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
    trainer.fit(diffuser, datamodule=datamodule)

    # ---------------- Execute Training ----------------
    trainer.fit(diffuser, datamodule=datamodule)
