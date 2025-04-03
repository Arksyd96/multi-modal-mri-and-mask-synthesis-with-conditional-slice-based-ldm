from typing import Dict, Tuple, Union, Optional
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb

from .base import DownEncoderBlock2D, UpDecoderBlock2D, UNetMidBlock2D
from modules.losses import LPIPS
from .unet_blocks import (
    BasicModel, UnetResBlock, UnetBasicBlock, DownBlock, BasicBlock, UpBlock
)

from pytorch_msssim import ssim

class ConditionMLP(nn.Module):
    def __init__(self, in_features, out_features, hidden_dim=512):
        super(ConditionMLP, self).__init__()
        
        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_features)
        )
        
    def forward(self, conditions):
        conditions = conditions.view(conditions.shape[0], -1)  # Flatten
        embedded_conditions = self.mlp(conditions)
        return embedded_conditions

class SinusoidalPosEmb(nn.Module):
    def __init__(self, emb_dim=16, downscale_freq_shift=1, max_period=1000, flip_sin_to_cos=False):
        super().__init__()
        self.emb_dim = emb_dim
        self.downscale_freq_shift = downscale_freq_shift
        self.max_period = max_period
        self.flip_sin_to_cos=flip_sin_to_cos

    def forward(self, x):
        device = x.device
        half_dim = self.emb_dim // 2
        emb = np.log(self.max_period) / (half_dim - self.downscale_freq_shift)
        emb = torch.exp(-emb * torch.arange(half_dim, device=device))
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)

        if self.flip_sin_to_cos:
            emb = torch.cat([emb[:, half_dim:], emb[:, :half_dim]], dim=-1)
        
        if self.emb_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
        return emb


class LearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with learned sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        half_dim = emb_dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim))

    def forward(self, x):
        x = x[:, None]
        freqs = x * self.weights[None, :] * 2 * np.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        if self.emb_dim % 2 == 1:
            fouriered = torch.nn.functional.pad(fouriered, (0, 1, 0, 0))
        return fouriered


class Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        temb_channels: int = 512,
        dropout: float = 0.0,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        double_z: bool = True,
        mid_block_add_attention=True
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[0],
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )

        self.mid_block = None
        self.down_blocks = nn.ModuleList([])

        # encoding blocks
        output_channel = block_out_channels[0]
        for i, block_out_channel in enumerate(block_out_channels):
            input_channel = output_channel
            output_channel = block_out_channel
            is_final_block = i == len(block_out_channels) - 1

            down_block = DownEncoderBlock2D(
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=temb_channels,
                dropout=dropout,
                num_layers=self.layers_per_block,
                resnet_time_scale_shift="scale_shift",
                resnet_groups=norm_num_groups,
                downsample=not is_final_block
            )
            self.down_blocks.append(down_block)

        # bottleneck
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=temb_channels,
            resnet_time_scale_shift="scale_shift",
            attention_num_heads=norm_num_groups,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # out
        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[-1], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()

        conv_out_channels = 2 * out_channels if double_z else out_channels
        self.conv_out = nn.Conv2d(block_out_channels[-1], conv_out_channels, 3, padding=1)

        self.gradient_checkpointing = True

    def forward(self, x: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        hidden_state = self.conv_in(x)

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # down
            for down_block in self.down_blocks:
                hidden_state = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(down_block), 
                    hidden_state, 
                    temb,
                    use_reentrant=False
                )
            # middle
            hidden_state = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block), 
                hidden_state, 
                temb,
                use_reentrant=False
            )
           
        else:
            for down_block in self.down_blocks:
                hidden_state = down_block(hidden_state, temb)

            # middle
            hidden_state = self.mid_block(hidden_state, temb)

        # post-process
        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        hidden_state = self.conv_out(hidden_state)

        return hidden_state


class Decoder(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        block_out_channels: Tuple[int, ...] = (64,),
        temb_channels: int = 512,
        layers_per_block: int = 2,
        norm_num_groups: int = 32,
        mid_block_add_attention=True
    ):
        super().__init__()
        self.layers_per_block = layers_per_block

        self.conv_in = nn.Conv2d(
            in_channels,
            block_out_channels[-1],
            kernel_size = 3,
            stride = 1,
            padding = 1,
        )

        self.mid_block = None
        self.up_blocks = nn.ModuleList([])

        # bottleneck
        self.mid_block = UNetMidBlock2D(
            in_channels=block_out_channels[-1],
            temb_channels=temb_channels,
            resnet_time_scale_shift="scale_shift",
            attention_num_heads=norm_num_groups,
            resnet_groups=norm_num_groups,
            add_attention=mid_block_add_attention,
        )

        # upsampling
        reversed_block_out_channels = list(reversed(block_out_channels))
        output_channel = reversed_block_out_channels[0]
        for i, reversed_block_out_channel in enumerate(reversed_block_out_channels):
            prev_output_channel = output_channel
            output_channel = reversed_block_out_channel

            is_final_block = i == len(reversed_block_out_channels) - 1

            up_block = UpDecoderBlock2D(
                in_channels=prev_output_channel,
                out_channels=output_channel,
                temb_channels=temb_channels,
                num_layers=self.layers_per_block + 1,
                resnet_time_scale_shift="scale_shift",
                upsample=not is_final_block
            )

            self.up_blocks.append(up_block)
            prev_output_channel = output_channel


        self.conv_norm_out = nn.GroupNorm(num_channels=block_out_channels[0], num_groups=norm_num_groups, eps=1e-6)
        self.conv_act = nn.SiLU()
        self.conv_out = nn.Conv2d(block_out_channels[0], out_channels, 3, padding=1)

        self.gradient_checkpointing = True

    def forward(self, hidden_state: torch.FloatTensor, latent_embeds: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        hidden_state = self.conv_in(hidden_state)

        upscale_dtype = next(iter(self.up_blocks.parameters())).dtype

        if self.training and self.gradient_checkpointing:
            def create_custom_forward(module):
                def custom_forward(*inputs):
                    return module(*inputs)

                return custom_forward

            # bottleneck
            hidden_state = torch.utils.checkpoint.checkpoint(
                create_custom_forward(self.mid_block),
                hidden_state,
                latent_embeds,
                use_reentrant=False,
            )
            hidden_state = hidden_state.to(upscale_dtype)

            # upsampling
            for up_block in self.up_blocks:
                hidden_state = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(up_block),
                    hidden_state,
                    latent_embeds,
                    use_reentrant=False,
                )

        else:
            # bottleneck
            hidden_state = self.mid_block(hidden_state, latent_embeds)
            hidden_state = hidden_state.to(upscale_dtype)

            # upsampling
            for up_block in self.up_blocks:
                hidden_state = up_block(hidden_state, latent_embeds)

        # post-process
        hidden_state = self.conv_norm_out(hidden_state)
        hidden_state = self.conv_act(hidden_state)
        x_hat = self.conv_out(hidden_state)

        return x_hat


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters: torch.Tensor, deterministic: bool = False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(
                self.mean, device=self.parameters.device, dtype=self.parameters.dtype
            )

    def sample(self) -> torch.FloatTensor:
        # make sure sample is on the same device as the parameters and has same dtype
        sample = torch.randn_like(
            self.mean,
            device=self.parameters.device,
            dtype=self.parameters.dtype,
        )
        x = self.mean + self.std * sample
        return x

    def kl(self, other: "DiagonalGaussianDistribution" = None) -> torch.FloatTensor:
        if self.deterministic:
            return torch.FloatTensor([0.0])
        else:
            if other is None:
                return 0.5 * torch.sum(
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar,
                    dim=[1, 2, 3],
                )
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var
                    - 1.0
                    - self.logvar
                    + other.logvar,
                    dim=[1, 2, 3],
                )

    def nll(self, sample: torch.Tensor, dims: Tuple[int, ...] = [1, 2, 3]) -> torch.FloatTensor:
        if self.deterministic:
            return torch.FloatTensor([0.0])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims,
        )

    def mode(self) -> torch.Tensor:
        return self.mean


class AutoencoderKL(pl.LightningModule):
    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        temb_channels: int = 128,
        max_period: int = 1000,
        block_out_channels: Tuple[int, ...] = (64, 128, 256, 512),
        layers_per_block: int = 2,
        latent_channels: int = 4,
        norm_num_groups: int = 32,
        pixel_loss_weight: float = 1.0,
        perceptual_loss_weight: float = 1.0,
        ssim_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.pixel_loss_weight = pixel_loss_weight
        self.perceptual_loss_weight = perceptual_loss_weight
        self.ssim_loss_weight = ssim_loss_weight
        self.kl_loss_weight = kl_loss_weight
        self.perceiver = LPIPS(linear_calibration=True).eval() if self.perceptual_loss_weight > 0 else None
        self.validation_steps = 0

        self.positional_encoder = None
        if temb_channels is not None:
            self.positional_encoder = nn.Sequential(
                SinusoidalPosEmb(emb_dim=temb_channels, max_period=max_period),
                nn.Linear(temb_channels, temb_channels * 4),
                nn.SiLU(),
                nn.Linear(temb_channels * 4, temb_channels)
            )

        self.encoder = Encoder(
            in_channels=in_channels,
            out_channels=latent_channels,
            block_out_channels=block_out_channels,
            temb_channels=temb_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            double_z=True,
            mid_block_add_attention=True
        )

        self.decoder = Decoder(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            temb_channels=temb_channels,
            layers_per_block=layers_per_block,
            norm_num_groups=norm_num_groups,
            mid_block_add_attention=True
        )

        self.quant_conv = nn.Conv2d(2 * latent_channels, 2 * latent_channels, 1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, 1)

        print(self.hparams)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, (Encoder, Decoder)):
            module.gradient_checkpointing = value

    def encode_position(self, position: torch.LongTensor) -> torch.FloatTensor:
        return self.positional_encoder(position) if self.positional_encoder is not None else None

    def encode(self, x: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> DiagonalGaussianDistribution:
        h = self.encoder(x, temb)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z: torch.FloatTensor, temb: Optional[torch.FloatTensor] = None) -> torch.FloatTensor:
        z = self.post_quant_conv(z)
        dec = self.decoder(z, temb)
        return dec

    def forward(
            self, x: torch.FloatTensor, position: torch.LongTensor = None, sample_posterior: bool = False, return_kl: bool = False
    ) -> torch.FloatTensor:
        temb = None
        if self.positional_encoder is not None:
            if position is None:
                raise ValueError("Positional encoding requires position tensor. ``None`` was provided.")
            temb = self.positional_encoder(position)

        posterior = self.encode(x, temb)

        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()

        dec = self.decode(z, temb)

        if return_kl:
            return dec, posterior.kl()
        return dec

    def perception_loss(self, predicted: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:
        if self.perceiver is not None:
            self.perceiver.eval()
            return self.perceiver(predicted, target) * self.perceptual_loss_weight
        return torch.FloatTensor([0.0])
    
    def ssim_loss(self, predicted: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:
        return 1 - ssim(
            ((predicted + 1) / 2).clamp(0, 1), 
            (target.type(predicted.dtype) + 1) / 2, data_range=1, size_average=False, nonnegative_ssim=True
        ).reshape(-1, *(1,) * (predicted.ndim - 1)) * self.ssim_loss_weight
    
    def reconstruction_loss(self, predicted: torch.FloatTensor, target: torch.FloatTensor) -> torch.FloatTensor:        
        # compute reconstruction loss
        perceptual_loss = self.perception_loss(predicted, target) # exclude seg channel
        ssim_loss = self.ssim_loss(predicted, target)
        pixel_loss = F.mse_loss(predicted, target, reduction='none') * self.pixel_loss_weight
        
        loss = torch.mean(perceptual_loss + ssim_loss + pixel_loss)
        return loss
    
    def training_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int) -> Dict[str, torch.FloatTensor]:
        return self.step(batch, batch_idx, split='train')
    
    def validation_step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int) -> Dict[str, torch.FloatTensor]:
        self.validation_steps = self.validation_steps + 1
        return self.step(batch, batch_idx, split='val')

    def step(self, batch: Tuple[torch.FloatTensor, torch.LongTensor], batch_idx: int, split: str = 'train') -> torch.FloatTensor:
        targets, positions = batch
        
        # ------------------------- Run Model ---------------------------
        predicted, kl = self.forward(targets, position=positions, sample_posterior=True, return_kl=True)

        # ------------------------- Compute Loss ---------------------------
        reconstruction_loss = self.reconstruction_loss(predicted, targets)
        regularization_loss = kl.mean() * self.kl_loss_weight
        loss = reconstruction_loss + regularization_loss
         
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss': loss, 'kullback': regularization_loss, 'reconstruction': reconstruction_loss}
            logging_dict['L2'] = torch.nn.functional.mse_loss(predicted, targets)
            logging_dict['L1'] = torch.nn.functional.l1_loss(predicted, targets)
            logging_dict['ssim'] = ssim((predicted + 1) / 2, (targets.type(predicted.dtype) + 1) / 2, data_range=1)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_value in logging_dict.items():
            self.log('{} - {}'.format(split, metric_name), metric_value, prog_bar=True,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )

        # -------------------------- Logging ---------------------------
        with torch.no_grad():
            if split == 'val' and (self.validation_steps + 1) % 80 == 0: 
                self.log_images(targets, predicted, label = 'Targets - Reconstructions')
    
        return loss

    def configure_optimizers(self):
        lr = 1e-05
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    @torch.no_grad()
    def log_images(self, *args, **kwargs):
        if self.trainer.global_rank == 0:
            # at this point x and x_hat are of shape [B, C, 128, 128]
            spatial_stack = lambda x: torch.cat([
                torch.hstack([img for img in x[:, idx, ...]]) for idx in range(x.shape[1])
            ], dim=0)

            args = [spatial_stack(arg) for arg in args]
            
            img = torch.cat(args, dim=0)

            # [-1, 1] => [0, 255]
            img = img.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
            
            wandb.log({
                'Reconstruction examples': wandb.Image(
                    img.detach().cpu().numpy(), 
                    caption='{} ({})'.format(self.trainer.global_step, kwargs['label'])
                )
            })


class VAE(BasicModel):
    def __init__(
        self,
        in_channels = 3, 
        out_channels = 3, 
        spatial_dims = 2,
        emb_channels = 4,
        hid_chs = [64, 128, 256, 512],
        kernel_sizes = [3, 3, 3, 3],
        strides = [1, 2, 2, 2],
        norm_name = ("GROUP", {'num_groups':8, "affine": True}),
        act_name = ("Swish", {}),
        temb_channels = None,
        max_period = 1000,
        dropout = None,
        use_res_block = True,
        deep_supervision = False,
        learnable_interpolation = True,
        use_attention = 'none',
        embedding_loss_weight = 1e-6,
        perceiver = LPIPS, 
        perceiver_kwargs = {},
        perceptual_loss_weight = 1.0,
        optimizer = torch.optim.Adam, 
        optimizer_kwargs = {'lr':1e-4},
        lr_scheduler = None, 
        lr_scheduler_kwargs = {},
        loss = torch.nn.L1Loss,
        loss_kwargs = {'reduction': 'none'},
        use_ssim_loss = True,
        use_perceptual_loss = True,
        sample_every_n_steps = 1000
    ):
        super().__init__(
            optimizer=optimizer,
            optimizer_kwargs=optimizer_kwargs,
            lr_scheduler=lr_scheduler,
            lr_scheduler_kwargs=lr_scheduler_kwargs
        )
        self.sample_every_n_steps=sample_every_n_steps
        self.loss_fct = loss(**loss_kwargs)
        self.use_ssim_loss = use_ssim_loss
        self.use_perceptual_loss = use_perceptual_loss
        self.embedding_loss_weight = embedding_loss_weight
        self.perceiver = perceiver(**perceiver_kwargs).eval() if perceiver is not None else None 
        self.perceptual_loss_weight = perceptual_loss_weight
        use_attention = use_attention if isinstance(use_attention, list) else [use_attention] * len(strides) 
        self.depth = len(strides)
        self.deep_supervision = deep_supervision
        downsample_kernel_sizes = kernel_sizes
        upsample_kernel_sizes = strides 

        self.save_hyperparameters()

        # -------- Time/Position embedding ---------
        self.positional_embedder = None
        if temb_channels is not None:
            self.positional_embedder = nn.Sequential(
                SinusoidalPosEmb(emb_dim=temb_channels, max_period=max_period),
                nn.Linear(temb_channels, temb_channels * 4),
                nn.SiLU(),
                nn.Linear(temb_channels * 4, temb_channels)
            )

        # ----------- In-Convolution ------------
        ConvBlock = UnetResBlock if use_res_block else UnetBasicBlock
        self.inc = ConvBlock(
            spatial_dims, 
            in_channels, 
            hid_chs[0], 
            kernel_size = kernel_sizes[0], 
            stride=strides[0],
            act_name = act_name, 
            norm_name = norm_name,
            emb_channels = temb_channels
        )

        # ----------- Encoder ----------------
        self.encoders = nn.ModuleList([
            DownBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i-1], 
                out_channels = hid_chs[i], 
                kernel_size = kernel_sizes[i], 
                stride = strides[i],
                downsample_kernel_size = downsample_kernel_sizes[i],
                norm_name = norm_name,
                act_name = act_name,
                dropout = dropout,
                use_res_block = use_res_block,
                learnable_interpolation = learnable_interpolation,
                use_attention = use_attention[i],
                emb_channels = temb_channels
            )
            for i in range(1, self.depth)
        ])

        # ----------- Out-Encoder ------------
        self.out_enc = nn.Sequential(
            BasicBlock(spatial_dims, hid_chs[-1], 2 * emb_channels, 3),
            BasicBlock(spatial_dims, 2 * emb_channels, 2 * emb_channels, 1)
        )

        # ----------- In-Decoder ------------
        self.inc_dec = ConvBlock(
            spatial_dims = spatial_dims,
            in_channels = emb_channels, 
            out_channels = hid_chs[-1], 
            kernel_size = 3, 
            act_name = act_name, 
            norm_name = norm_name, 
            emb_channels = temb_channels
        ) 

        # ------------ Decoder ----------
        self.decoders = nn.ModuleList([
            UpBlock(
                spatial_dims = spatial_dims, 
                in_channels = hid_chs[i+1], 
                out_channels = hid_chs[i],
                kernel_size = kernel_sizes[i+1], 
                stride = strides[i+1], 
                upsample_kernel_size = upsample_kernel_sizes[i+1],
                norm_name = norm_name,  
                act_name = act_name, 
                dropout = dropout,
                use_res_block = use_res_block,
                learnable_interpolation = learnable_interpolation,
                use_attention = use_attention[i],
                emb_channels = temb_channels,
                skip_channels = 0
            ) for i in range(self.depth - 1)
        ])

        # --------------- Out-Convolution ----------------
        self.outc = BasicBlock(spatial_dims, hid_chs[0], out_channels, 1, zero_conv=True)
        
        # if isinstance(deep_supervision, bool):
        deep_supervision = self.depth - 1 if deep_supervision else 0 
            
        self.outc_ver = nn.ModuleList([
            BasicBlock(spatial_dims, hid_chs[i], out_channels, 1, zero_conv=True) 
            for i in range(1, deep_supervision + 1)
        ])

        self.save_hyperparameters()
        
    def encode_position(self, position):
        if self.positional_embedder is not None:
            return self.positional_embedder(position)
        return None
    
    def encode(self, x, emb=None, return_posterior=False):
        h = self.inc(x, emb=emb)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h, emb=emb)
        moments = self.out_enc(h)
        posterior = DiagonalGaussianDistribution(moments)
        if return_posterior:
            return posterior
        return posterior.sample()
            
    def decode(self, z, emb=None):
        h = self.inc_dec(z, emb=emb)
        for i in range(len(self.decoders), 0, -1):
            h = self.decoders[i-1](h, emb=emb)
        x = self.outc(h)
        return x 

    def forward(self, x_in, position=None, return_kl=False):
        # --------- Time Embedding -----------
        emb = self.encode_position(position)

        # --------- Encoder --------------
        h = self.inc(x_in, emb=emb)
        for i in range(len(self.encoders)):
            h = self.encoders[i](h, emb=emb)
        moments = self.out_enc(h)

        # --------- Quantizer --------------
        posterior = DiagonalGaussianDistribution(moments)
        z_q, emb_loss = posterior.sample(), posterior.kl()

        # -------- Decoder -----------
        h = self.inc_dec(z_q, emb=emb)
        for i in range(len(self.decoders) - 1, -1, -1):
            h = self.decoders[i](h, emb=emb)
        out = self.outc(h)

        if return_kl:
            return out, emb_loss
        return out
    
    def perception_loss(self, prediction, target):
        if self.perceiver is not None:
            self.perceiver.eval()
            return self.perceiver(prediction, target) * self.perceptual_loss_weight
        return torch.FloatTensor([0.0])
    
    def ssim_loss(self, pred, target):
        return 1 - ssim(
            ((pred + 1) / 2).clamp(0, 1), 
            (target.type(pred.dtype) + 1) / 2, data_range=1, size_average=False, nonnegative_ssim=True
        ).reshape(-1, *(1,) * (pred.ndim - 1))
    
    def rec_loss(self, pred, target):        
        # compute reconstruction loss
        perceptual_loss = self.perception_loss(pred, target) if self.use_perceptual_loss else 0
        ssim_loss = self.ssim_loss(pred, target) if self.use_ssim_loss else 0
        pixel_loss = self.loss_fct(pred, target)

        loss = torch.mean(perceptual_loss + ssim_loss + pixel_loss)

        return loss
    
    def _step(self, batch, batch_idx, split, step):
        # ------------------------- Get Source/Target ---------------------------
        target = batch[0]
        position = batch[1] if len(batch) > 1 else None
        position = position.squeeze(1) if position is not None else None
        
        if self.positional_embedder is None:
            position = None
        
        # ------------------------- Run Model ---------------------------
        prediction, emb_loss = self.forward(target, position=position, return_kl=True)

        # ------------------------- Compute Loss ---------------------------
        loss = self.rec_loss(prediction, target)
        loss = loss + emb_loss.mean() * self.embedding_loss_weight
         
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            logging_dict = {'loss': loss, 'emb_loss': emb_loss.mean()}
            logging_dict['L2'] = torch.nn.functional.mse_loss(prediction, target)
            logging_dict['L1'] = torch.nn.functional.l1_loss(prediction, target)    
            logging_dict['ssim'] = ssim((prediction + 1) / 2, (target.type(prediction.dtype) + 1) / 2, data_range=1)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in logging_dict.items():
            self.log('{} - {}'.format(split, metric_name), metric_val, prog_bar=True,
                on_step=True, on_epoch=True, sync_dist=True, logger=True
            )

        # -------------------------- Logging ---------------------------
        with torch.no_grad():
            if split == 'val' and (self._step_val + 1) % self.sample_every_n_steps == 0: 
                self.log_images(target, prediction, label = 'original - reconstructed')
    
        return loss
    
    def log_images(self, *args, **kwargs):
        if self.trainer.global_rank == 0:
            # at this point x and x_hat are of shape [B, C, 128, 128]
            spatial_stack = lambda x: torch.cat([
                torch.hstack([img for img in x[:, idx, ...]]) for idx in range(x.shape[1])
            ], dim=0)

            # => if 3D will be of shape B, 3, 128, 128, 64
            if args[0].ndim == 5:
                rand_idx = np.random.randint(0, args[0].shape[-1], size=(4,))
                args = [arg[..., rand_idx].permute(4, 0, 1, 2, 3) for arg in args]
                args = [arg.reshape(-1, *arg.shape[2:]) for arg in args]

            args = [spatial_stack(arg) for arg in args]
            
            img = torch.cat(args, dim=0)

            # [-1, 1] => [0, 255]
            img = img.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)
            
            wandb.log({
                'Reconstruction examples': wandb.Image(
                    img.detach().cpu().numpy(), 
                    caption='{} ({})'.format(self.trainer.global_step, kwargs['label'])
                )
            })