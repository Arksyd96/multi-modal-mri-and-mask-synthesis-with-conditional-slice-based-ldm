import numpy as np
import torch
import pytorch_lightning as pl
import os
from omegaconf import OmegaConf
from pytorch_lightning.loggers import wandb as wandb_logger
from pytorch_lightning.callbacks import ModelCheckpoint

from modules.data_preprocessing import SegAugDataModule
from monai.networks.nets import UNet, VNet, DynUNet, UNETR, SwinUNETR, AttentionUnet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric, hausdorff_distance

os.environ['WANDB_API_KEY'] = 'bdc8857f9d6f7010cff35bcdc0ae9413e05c75e1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def global_seed(seed, debugging=False):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    if debugging:
        torch.backends.cudnn.deterministic = True

class LitUNet(pl.LightningModule):
    def __init__(self, 
        spatial_dims, 
        in_channels, 
        out_channels,
        channels = (16, 32, 64, 128, 256),
        strides = (2, 2, 2, 2),
        kernel_size = 3,
        lr = 1e-5
    ):
        super().__init__()
        self.backbone = SwinUNETR(
            img_size=(128, 128, 64),
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_dims=3
        )

        self.lr = lr
        self.loss = DiceLoss(sigmoid=True)
        self.metric = DiceMetric(include_background=False, reduction='mean', num_classes=2)
        self.save_hyperparameters()

        print('Using {} as architecture'.format(self.backbone.__class__.__name__))

    def forward(self, x):
        return self.backbone(x)  

    def dice_score_per_label(self, mask_hat, mask):
        dice_score = self.metric(mask_hat, mask) # [B, C]
        mean_dice_score = torch.mean(dice_score, dim=0)
        return mean_dice_score
                 
    def training_step(self, batch, batch_idx):
        x, mask = batch[0][:, :-1], batch[0][:, -1, None]
        mask[mask == -1] = 0
        mask_hat = self.forward(x)
        loss = self.loss(mask_hat, mask)

        # logging
        self.log('dice_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, mask = batch[0][:, :-1], batch[0][:, -1, None]
        mask[mask == -1] = 0
        mask_hat = self.forward(x)
        loss = self.loss(mask_hat, mask)

        # logging
        self.log('val_dice_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        
        return loss
    
    def test_step(self, batch, batch_idx):
        x, mask = batch[0][:, :-1], batch[0][:, -1, None]
        mask[mask == -1] = 0
        mask_hat = self.forward(x)
        loss = self.loss(mask_hat, mask)
        dice = self.dice_score_per_label(torch.sigmoid(mask_hat).round(), mask)

        # logging
        self.log('test_dice_loss', loss, on_step=True, on_epoch=True, logger=True, sync_dist=True)
        self.log('test_dice_score', dice[0], on_epoch=True, logger=True, sync_dist=True)
        
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
if __name__ == "__main__":
    global_seed(42)
    torch.set_float32_matmul_precision('high')
    
    n_reals, n_fakes = 100, 200

    # logger
    # logger = wandb_logger.WandbLogger(
    #     project='improved-slice-based-latent-diffusion-model', 
    #     name='Training segmentation (SBLDM) ({}/{} real/fake)'.format(n_reals, n_fakes)
    # )

    # defining unet's architecture
    model = LitUNet(spatial_dims=3, in_channels=2, out_channels=1, lr=3e-4) # outputs 4 classes (background, C1, C2, C3)
    
    # data module
    datamodule = SegAugDataModule(
        real_dir = './data/second_stage_dataset_128x128.npy',
        fake_dir = './samples/latest_sbldm_samples_cleaned.npy',
        classes = [1, 3],
        n_reals = n_reals,
        n_fakes = n_fakes,
        batch_size = 4,
        num_workers = 16,
        dtype = 'float32',
        shuffle = True,
        vertical_flip=0.66,
        horizontal_flip=0.66,
        rotation=[0, 180]
    )
    
    # # callbacks
    # checkpoint_callback = ModelCheckpoint(
    #     **cfg.callbacks.checkpoint,
    #     filename='UNet-ckpt-{epoch}',
    # )
    
    # training
    trainer = pl.Trainer(
        # logger=logger,
        # strategy="ddp",
        # devices=4,
        # num_nodes=2,
        accelerator='gpu',
        precision=32,
        max_epochs=100,
        log_every_n_steps=1,
        check_val_every_n_epoch=10,
        enable_progress_bar=True,
        enable_checkpointing=False,
        num_sanity_val_steps=0
        # callbacks=[checkpoint_callback]
    )

    trainer.fit(model=model, datamodule=datamodule)
    # trainer.test(model=model, datamodule=datamodule)
    
    
        
    
