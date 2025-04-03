"""
    Code inspired by medfusion public repo : https://github.com/mueller-franzes/medfusion
    And official Open AI repo : https://github.com/openai/guided-diffusion (https://arxiv.org/abs/2212.07501)
    The code is modified to fit the needs of the project
"""
from typing import Union, Tuple, Any
from tqdm import tqdm

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import pytorch_lightning as pl
import wandb

# from modules.models.base import BasicModel
from modules.ema import EMAModel # TODO
from modules.models.autoencoder import AutoencoderKL
from modules.utils import kl_gaussians # TODO
# from diffusers import DDPMScheduler, DDIMScheduler, UNet2DModel, LDMPipeline
from modules.models.unet import UNet
from modules.models.unet_blocks import BasicModel


class DiffusionPipeline(BasicModel):
    def __init__(self, 
        noise_scheduler,
        noise_estimator,
        latent_embedder = None,
        estimator_objective = 'x_T', # 'x_T' or 'x_0'
        estimate_variance = False, 
        use_self_conditioning = False, 
        classifier_free_guidance_dropout = 0.5, # Probability to drop condition during training, has only an effect for label-conditioned training 
        clip_x0 = False, # Has only an effect during traing if use_self_conditioning=True, import for inference/sampling  
        use_ema = False,
        ema_kwargs = {},
        optimizer = torch.optim.AdamW, 
        optimizer_kwargs = {'lr': 1e-4}, # stable-diffusion ~ 1e-4
        lr_scheduler = None, # stable-diffusion - LambdaLR
        lr_scheduler_kwargs = {}, 
        loss = torch.nn.L1Loss,
        loss_kwargs = {},
        std_norm = None,
        sample_every_n_steps = 500,
        slice_based = False
    ):
        super().__init__(optimizer, optimizer_kwargs, lr_scheduler, lr_scheduler_kwargs)
        self.loss_fct = loss(**loss_kwargs)
        self.sample_every_n_steps = sample_every_n_steps

        self.noise_scheduler = noise_scheduler
        self.noise_estimator = noise_estimator
        self.latent_embedder = latent_embedder
        if self.latent_embedder is not None:
            self.latent_embedder.freeze()
            self.latent_embedder.requires_grad_(False)

        self.estimator_objective = estimator_objective
        self.use_self_conditioning = use_self_conditioning
        self.classifier_free_guidance_dropout = classifier_free_guidance_dropout
        self.estimate_variance = estimate_variance
        self.clip_x0 = clip_x0
        self.std_norm = std_norm
        self.slice_based = slice_based

        self.use_ema = use_ema
        if use_ema:
            self.ema_model = EMAModel(self.noise_estimator, **ema_kwargs)

        self.save_hyperparameters(ignore=['latent_embedder'])
    
    @torch.no_grad()
    def encode_targets(self, targets: torch.FloatTensor) -> torch.FloatTensor:
        # x should be of shape [B, C, W, H, D]
        positional_embeddings = self.latent_embedder.encode_position(
            torch.arange(0, self.latent_embedder.hparams.max_period)
            .to(targets.device, dtype=self.latent_embedder.dtype)
        )

        latents = []
        for x_idx in range(targets.shape[0]):
            curr_target = targets[x_idx].permute(3, 0, 1, 2) # => [D, C, W, H]
            latents.append(
                self.latent_embedder.encode(curr_target, positional_embeddings).sample()
            )
        latents = torch.stack(latents, dim=0)
        return latents
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.FloatTensor, channels_first: bool = False) -> torch.FloatTensor:
        # latents should be of shape [B, C, W, H, D] or [B, D, C, W, H]
        positional_embeddings = self.latent_embedder.encode_position(
            torch.arange(0, self.latent_embedder.hparams.max_period)
            .to(latents.device, dtype=self.latent_embedder.dtype)
        )

        volumes = []
        for lat_idx in range(latents.shape[0]):
            curr_latents = latents[lat_idx]
            if channels_first:
                curr_latents = curr_latents.permute(3, 0, 1, 2) # [C, W, H, D] => [D, C, W, H]
            volumes.append(self.latent_embedder.decode(curr_latents, positional_embeddings))

        volumes = torch.stack(volumes, dim=0).permute(0, 2, 3, 4, 1) # => [B, D, C, W, H]
        return volumes

    def _step(self, batch, batch_idx, state, step):
        results = {}
        x_0 = batch[0]
        condition = batch[1] if len(batch) > 1 else None

        if self.latent_embedder is not None:
            # Embed into latent space or normalize 
            if self.slice_based:
                x_0 = self.encode_targets(x_0)
                x_0 = x_0.permute(0, 2, 3, 4, 1) # => [B, 2, 16, 16, 64]
                # x_0 = x_0.reshape(x_0.shape[0], self.noise_estimator.in_channels, *self.noise_estimator.sample_size)

            else:
                x_0 = self.latent_embedder.encode(x_0, None)


        if self.std_norm is not None:
            x_0 = x_0.div(self.std_norm)

        if self.clip_x0:
            x_0 = torch.clamp(x_0, -1, 1)

        # Sample Noise
        with torch.no_grad():
            # Randomly selecting t [0,T-1] and compute x_t (noisy version of x_0 at t)
            x_t, x_T, t = self.noise_scheduler.sample(x_0)
                
        # Use EMA Model
        if self.use_ema and (state != 'train'):
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Re-estimate x_T or x_0, self-conditioned on previous estimate 
        self_cond = None 
        if self.use_self_conditioning:
            with torch.no_grad():
                pred, pred_vertical = noise_estimator(x_t, t, condition, None) 
                if self.estimate_variance:
                    pred, _ =  pred.chunk(2, dim = 1)  # Seperate actual prediction and variance estimation 
                if self.estimator_objective == "x_T": # self condition on x_0 
                    self_cond = self.noise_scheduler.estimate_x_0(x_t, pred, t=t, clip_x0=self.clip_x0)
                elif self.estimator_objective == "x_0": # self condition on x_T 
                    self_cond = self.noise_scheduler.estimate_x_T(x_t, pred, t=t, clip_x0=self.clip_x0)
                else:
                    raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")
            
        # Classifier free guidance 
        if torch.rand(1) <= self.classifier_free_guidance_dropout:
            condition = None 
       
        # Run Denoise 
        # pred, pred_vertical = noise_estimator(x_t, t, condition, self_cond)
        pred = noise_estimator(x_t, t, condition)
        pred_vertical = []
        
        # Separate variance (scale) if it was learned 
        if self.estimate_variance:
            pred, pred_var = pred.chunk(2, dim = 1)  # Separate actual prediction and variance estimation 

        # Specify target 
        if self.estimator_objective == "x_T":
            target = x_T 
        elif self.estimator_objective == "x_0":
            target = x_0 
        else:
            raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")
        
        # ------------------------- Compute Loss ---------------------------
        interpolation_mode = 'area'
        loss = 0
        weights = [1 / 2 ** i for i in range(1 + len(pred_vertical))] # horizontal (equal) + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w / tot_weight for w in weights]

        # ----------------- MSE/L1, ... ----------------------
        loss += self.loss_fct(pred, target) * weights[0]

        # ----------------- Variance Loss --------------
        if self.estimate_variance:
            # var_scale = var_scale.clamp(-1, 1) # Should not be necessary 
            var_scale = (pred_var + 1) / 2 # Assumed to be in [-1, 1] -> [0, 1] 
            pred_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=var_scale)
            # pred_logvar = pred_var  # If variance is estimated directly 

            if  self.estimator_objective == 'x_T':
                pred_x_0 = self.noise_scheduler.estimate_x_0(x_t, x_T, t, clip_x0=self.clip_x0)
            elif self.estimator_objective == "x_0":
                pred_x_0 = pred 
            else:
                raise NotImplementedError()

            with torch.no_grad():
                pred_mean = self.noise_scheduler.estimate_mean_t(x_t, pred_x_0, t)
                true_mean = self.noise_scheduler.estimate_mean_t(x_t, x_0, t)
                true_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=0)
            
            kl_loss = torch.mean(kl_gaussians(true_mean, true_logvar, pred_mean, pred_logvar), dim=list(range(1, x_0.ndim)))
            nnl_loss = torch.mean(F.gaussian_nll_loss(pred_x_0, x_0, torch.exp(pred_logvar), reduction='none'), dim=list(range(1, x_0.ndim)))
            var_loss = torch.mean(torch.where(t == 0, nnl_loss, kl_loss))
            loss += var_loss
            
            results['variance_scale'] = torch.mean(var_scale)
            results['variance_loss'] = var_loss

            
        # ----------------------------- Deep Supervision -------------------------
        for i, pred_i in enumerate(pred_vertical): 
            target_i = F.interpolate(target, size=pred_i.shape[2:], mode=interpolation_mode, align_corners=None)  
            loss += self.loss_fct(pred_i, target_i)*weights[i+1]
        results['loss']  = loss
       
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            results['L2'] = F.mse_loss(pred, target)
            results['L1'] = F.l1_loss(pred, target)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(f"{state} - {metric_name}", metric_val, batch_size=x_0.shape[0], on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        
        return loss
    

    def forward(self, x_t, t, condition=None, self_cond=None, guidance_scale=1.0, cold_diffusion=False, un_cond=None):
        # Note: x_t expected to be in range ~ [-1, 1]
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Concatenate inputs for guided and unguided diffusion as proposed by classifier-free-guidance
        if (condition is not None) and (guidance_scale != 1.0):
            # Model prediction 
            pred_uncond = noise_estimator(x_t, t, condition=un_cond, self_cond=self_cond)
            pred_cond = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond =  pred_uncond.chunk(2, dim = 1)  
                pred_cond,   pred_var_cond =  pred_cond.chunk(2, dim = 1) 
                pred_var = pred_var_uncond + guidance_scale * (pred_var_cond - pred_var_uncond)
        else:
            pred =  noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            if self.estimate_variance:
                pred, pred_var =  pred.chunk(2, dim = 1)  

        if self.estimate_variance:
            pred_var_scale = pred_var / 2 + 0.5 # [-1, 1] -> [0, 1]
            pred_var_value = pred_var  
        else:
            pred_var_scale = 0
            pred_var_value = None 

        # pred_var_scale = pred_var_scale.clamp(0, 1)

        if  self.estimator_objective == 'x_0':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(
                x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion
            )
            x_T = self.noise_scheduler.estimate_x_T(x_t, x_0=pred, t=t, clip_x0=self.clip_x0)
            self_cond = x_T 
        elif self.estimator_objective == 'x_T':
            if x_t.shape[1] != pred.shape[1]:
                x_t = x_t[:, :pred.shape[1]]
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(
                x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion
            )
            x_T = pred 
            self_cond = x_0 
        else:
            raise ValueError("Unknown Objective")
        
        return x_t_prior, x_0, x_T, self_cond 


    @torch.no_grad()
    def denoise(self, x_t, steps=None, condition=None, use_ddim=False, **kwargs):
        self_cond = None 

        # ---------- run denoise loop ---------------
        steps = self.noise_scheduler.timesteps if steps is None else steps
        if use_ddim:
            timesteps_array = torch.linspace(0, self.noise_scheduler.T-1, steps, dtype=torch.long, device=x_t.device) # [0, 1, 2, ..., T-1] if steps = T 
        else:
            timesteps_array = self.noise_scheduler.timesteps_array[slice(0, steps)] # [0, ...,T-1] (target time not time of x_t)
            
        for i, t in tqdm(enumerate(reversed(timesteps_array))):
            # UNet prediction
            x_t, x_0, x_T, self_cond = self.forward(x_t, t.expand(x_t.shape[0]), condition, self_cond=self_cond, **kwargs)
            self_cond = self_cond if self.use_self_conditioning else None  
        
            if use_ddim and (steps-i-1>0):
                t_next = timesteps_array[steps-i-2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = kwargs.get('eta', 1) * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise

        # ------ Eventually decode from latent space into image space--------
        if self.latent_embedder is not None:
            x_t = self.latent_embedder.decode(x_t)
        
        return x_t # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(self, num_samples, img_size, condition=None, steps=None, use_ddim=False, **kwargs):
        template = torch.zeros((num_samples, *img_size), device=self.device)
        x_T = self.noise_scheduler.x_final(template)
        x_0 = self.denoise(
            x_T, 
            steps = steps, 
            condition = condition.to(self.device, dtype=self.dtype) if condition is not None else None, 
            use_ddim = use_ddim,
            **kwargs
        )
        return x_0 
    

    @torch.no_grad()
    def interpolate(self, img1, img2, i = None, condition=None, lam = 0.5, **kwargs):
        assert img1.shape == img2.shape, "Image 1 and 2 must have equal shape"

        t = self.noise_scheduler.T-1 if i is None else i
        t = torch.full(img1.shape[:1], i, device=img1.device)

        img1_t = self.noise_scheduler.estimate_x_t(img1, t=t, clip_x0=self.clip_x0)
        img2_t = self.noise_scheduler.estimate_x_t(img2, t=t, clip_x0=self.clip_x0)

        img = (1 - lam) * img1_t + lam * img2_t
        img = self.denoise(img, i, condition, **kwargs)
        return img

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.step(self.noise_estimator)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.noise_estimator.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]


def pad_to_64x64(tensor):
    # Calculate the amount of padding needed
    pad_height = max(0, 64 - tensor.shape[1])
    pad_width = max(0, 64 - tensor.shape[2])

    # Pad the tensor with -1 values
    padded_tensor = F.pad(tensor, (0, pad_width, 0, pad_height), value=-1)

    return padded_tensor
        

class TumorEnhancingDiffusionPipeline(DiffusionPipeline):
    def __init__(self, 
        noise_scheduler,
        noise_estimator,
        latent_embedder: AutoencoderKL,
        estimator_objective = 'x_T', # 'x_T' or 'x_0'
        estimate_variance = False, 
        use_self_conditioning = False, 
        classifier_free_guidance_dropout = 0.5,
        clip_x0 = False, # Has only an effect during traing if use_self_conditioning=True, import for inference/sampling  
        use_ema = False,
        ema_kwargs = {},
        optimizer = torch.optim.AdamW, 
        optimizer_kwargs = {'lr':1e-4}, # stable-diffusion ~ 1e-4
        lr_scheduler = None, # stable-diffusion - LambdaLR
        lr_scheduler_kwargs = {}, 
        loss = torch.nn.L1Loss,
        loss_kwargs = {},
        sample_every_n_steps = 500
    ):
        super().__init__(
            noise_scheduler, 
            noise_estimator, 
            latent_embedder, 
            estimator_objective, 
            estimate_variance, 
            use_self_conditioning, 
            classifier_free_guidance_dropout, 
            clip_x0, 
            use_ema, 
            ema_kwargs, 
            optimizer, 
            optimizer_kwargs, 
            lr_scheduler, 
            lr_scheduler_kwargs, 
            loss, 
            loss_kwargs, 
            None, 
            sample_every_n_steps, 
            False
        )

    
    @torch.no_grad()
    def encode_targets(self, targets: torch.FloatTensor) -> torch.FloatTensor:
        return targets
    
    @torch.no_grad()
    def decode_latents(self, latents: torch.FloatTensor, channels_first: bool = False) -> torch.FloatTensor:
        return latents
    
    def _extract_tumor_patch(self, sequence, binary_mask, window_size=(64, 64)):
        if binary_mask.sum() == 0:
            raise ValueError('Binary mask is empty')
        
        tumor_patch = list()
        sequence = (sequence.add(1).div(2) * binary_mask).mul(2).sub(1)

        for idx in range(sequence.shape[0]):
            # Find the bounding box of the tumor mask
            nonzero_indices = torch.nonzero(binary_mask[idx].squeeze(0))
            min_x = nonzero_indices[:, 0].min()
            max_x = nonzero_indices[:, 0].max()
            min_y = nonzero_indices[:, 1].min()
            max_y = nonzero_indices[:, 1].max()

            # Calculate center coordinates
            center_x = (min_x + max_x) // 2
            center_y = (min_y + max_y) // 2

            # Calculate offsets to extract a 64 x 64 patch around the tumor center
            offset_x = max(0, center_x - window_size[0] // 2)
            offset_y = max(0, center_y - window_size[1] // 2)

            # Extract patches around the tumor center from flair and t1ce images
            tumor_patch.append(pad_to_64x64(
                sequence[
                    idx,
                    :, 
                    offset_x:offset_x + window_size[0], 
                    offset_y:offset_y + window_size[1]
                ]
            ))

        return torch.stack(tumor_patch, dim=0)

    def _step(self, batch: Tuple[torch.FloatTensor, ...], batch_idx: int, state: str, step: int):
        results = {}
        hr_image = batch[0]
        position = batch[1] if batch.__len__() > 1 else None

        condition = None # condition is made channel-wise

        # extracting the plain binary mask
        binary_mask = (hr_image[:, -1, None, ...] > -1).type(hr_image.dtype)

        if self.latent_embedder is not None:
            if position is not None:
                lr_image = self.latent_embedder.forward(
                    torch.cat([hr_image[:, :-1, ...], binary_mask.mul(2).sub(1)], dim=1), # hr_image with a binary mask 
                    position.squeeze(1), 
                    sample_posterior=True
                )
            else:
                raise ValueError("Position is required when using latent embedder")
            
        # droping masks
        lr_image = lr_image[:, :-1, ...].detach()

        lr_tumor_patch = self._extract_tumor_patch(lr_image, binary_mask)
        hr_tumor_patch = self._extract_tumor_patch(hr_image, binary_mask)

        if self.clip_x0:
            lr_tumor_patch = torch.clamp(lr_tumor_patch, -1, 1)
            hr_tumor_patch = torch.clamp(hr_tumor_patch, -1, 1)

        with torch.no_grad():
            # adding noise to the high resolution image only
            x_t, x_T, t = self.noise_scheduler.sample(hr_tumor_patch)

        # concatenating the low res to the high res
        x_t = torch.cat([x_t, lr_tumor_patch], dim=1) # condition is the second half of the tensor
                
        # Use EMA Model
        if self.use_ema and (state != 'train'):
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Re-estimate x_T or x_0, self-conditioned on previous estimate 
        self_cond = None 
        if self.use_self_conditioning:
            with torch.no_grad():
                pred, pred_vertical = noise_estimator(x_t, t, condition, None) 
                if self.estimate_variance:
                    pred, _ =  pred.chunk(2, dim = 1)  # Seperate actual prediction and variance estimation 
                if self.estimator_objective == "x_T": # self condition on x_0 
                    self_cond = self.noise_scheduler.estimate_x_0(x_t, pred, t=t, clip_x0=self.clip_x0)
                elif self.estimator_objective == "x_0": # self condition on x_T 
                    self_cond = self.noise_scheduler.estimate_x_T(x_t, pred, t=t, clip_x0=self.clip_x0)
                else:
                    raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")
            
        # Classifier free guidance 
        if torch.rand(1) <= self.classifier_free_guidance_dropout:
            condition = None 
       
        # Run Denoise 
        # pred, pred_vertical = noise_estimator(x_t, t, condition, self_cond)
        pred = noise_estimator(x_t, t, condition)
        pred_vertical = []
        
        # Separate variance (scale) if it was learned 
        if self.estimate_variance:
            pred, pred_var = pred.chunk(2, dim = 1)  # Separate actual prediction and variance estimation 

        # Specify target 
        if self.estimator_objective == "x_T":
            target = x_T # predicting noise
        elif self.estimator_objective == "x_0":
            target = hr_image # predicting the original hr image
        else:
            raise NotImplementedError(f"Option estimator_target={self.estimator_objective} not supported.")
        
        # ------------------------- Compute Loss ---------------------------
        interpolation_mode = 'area'
        loss = 0
        weights = [1 / 2 ** i for i in range(1 + len(pred_vertical))] # horizontal (equal) + vertical (reducing with every step down)
        tot_weight = sum(weights)
        weights = [w / tot_weight for w in weights]

        # ----------------- MSE/L1, ... ----------------------
        loss += self.loss_fct(pred, target) * weights[0]

        # ----------------- Variance Loss --------------
        if self.estimate_variance:
            # var_scale = var_scale.clamp(-1, 1) # Should not be necessary 
            var_scale = (pred_var + 1) / 2 # Assumed to be in [-1, 1] -> [0, 1] 
            pred_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=var_scale)
            # pred_logvar = pred_var  # If variance is estimated directly 

            if  self.estimator_objective == 'x_T':
                pred_x_0 = self.noise_scheduler.estimate_x_0(x_t, x_T, t, clip_x0=self.clip_x0)
            elif self.estimator_objective == "x_0":
                pred_x_0 = pred 
            else:
                raise NotImplementedError()

            with torch.no_grad():
                pred_mean = self.noise_scheduler.estimate_mean_t(x_t, pred_x_0, t)
                true_mean = self.noise_scheduler.estimate_mean_t(x_t, hr_image, t)
                true_logvar = self.noise_scheduler.estimate_variance_t(t, x_t.ndim, log=True, var_scale=0)
            
            kl_loss = torch.mean(kl_gaussians(true_mean, true_logvar, pred_mean, pred_logvar), dim=list(range(1, hr_image.ndim)))
            nnl_loss = torch.mean(F.gaussian_nll_loss(pred_x_0, hr_image, torch.exp(pred_logvar), reduction='none'), dim=list(range(1, hr_image.ndim)))
            var_loss = torch.mean(torch.where(t == 0, nnl_loss, kl_loss))
            loss += var_loss
            
            results['variance_scale'] = torch.mean(var_scale)
            results['variance_loss'] = var_loss
       
        # --------------------- Compute Metrics  -------------------------------
        with torch.no_grad():
            results['L2'] = F.mse_loss(pred, target)
            results['L1'] = F.l1_loss(pred, target)

        # ----------------- Log Scalars ----------------------
        for metric_name, metric_val in results.items():
            self.log(
                f"{state} - {metric_name}", 
                metric_val, 
                batch_size = hr_image.shape[0], 
                on_step=True, on_epoch=True, sync_dist=True, prog_bar=True
            )
        
        if (self.global_step + 1) % self.sample_every_n_steps == 0:
            self.log_samples(lr_sample = lr_tumor_patch)
        
        return loss
    
    def log_samples(self, lr_sample: torch.FloatTensor, *kwargs: Tuple[Any, ...]):
        if self.trainer.global_rank == 0:
            hr_sample = self.sample(
                lr_sample, 
                steps = self.noise_scheduler.timesteps,
                condition = None,
                use_ddim = False
            ) # => [B, estimator.in_channels, W, D] (without condition)

            spatial_stack = lambda x: torch.cat([
                torch.hstack([img for img in x[:, idx, ...]]) 
                for idx in range(x.shape[1])
            ], dim = 0)

            sample = spatial_stack(torch.cat([lr_sample, hr_sample], dim=1))
            sample = sample.add(1).div(2).mul(255).clamp(0, 255).to(torch.uint8)

            wandb.log({
                'Super resolution examples': wandb.Image(
                    sample.detach().cpu().numpy(), 
                    caption='({} - LR / HR)'.format(self.trainer.global_step)
                )
            })

    
    def forward(self, x_t, t, condition=None, self_cond=None, guidance_scale=1.0, cold_diffusion=False, un_cond=None):
        # Note: x_t expected to be in range ~ [-1, 1]
        if self.use_ema:
            noise_estimator = self.ema_model.averaged_model
        else:
            noise_estimator = self.noise_estimator

        # Concatenate inputs for guided and unguided diffusion as proposed by classifier-free-guidance
        if (condition is not None) and (guidance_scale != 1.0):
            # Model prediction 
            pred_uncond = noise_estimator(x_t, t, condition=un_cond, self_cond=self_cond)
            pred_cond = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)

            if self.estimate_variance:
                pred_uncond, pred_var_uncond =  pred_uncond.chunk(2, dim = 1)  
                pred_cond,   pred_var_cond =  pred_cond.chunk(2, dim = 1) 
                pred_var = pred_var_uncond + guidance_scale * (pred_var_cond - pred_var_uncond)
        else:
            pred = noise_estimator(x_t, t, condition=condition, self_cond=self_cond)
            if self.estimate_variance:
                pred, pred_var =  pred.chunk(2, dim = 1)  

        if self.estimate_variance:
            pred_var_scale = pred_var / 2 + 0.5 # [-1, 1] -> [0, 1]
            pred_var_value = pred_var  
        else:
            pred_var_scale = 0
            pred_var_value = None 

        # pred_var_scale = pred_var_scale.clamp(0, 1)

        if  self.estimator_objective == 'x_0':
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_0(
                x_t, t, pred, clip_x0=self.clip_x0, var_scale=pred_var_scale, cold_diffusion=cold_diffusion
            )
            x_T = self.noise_scheduler.estimate_x_T(x_t, x_0=pred, t=t, clip_x0=self.clip_x0)
            self_cond = x_T 
        elif self.estimator_objective == 'x_T':
            if x_t.shape[1] != pred.shape[1]:
                x_t = x_t[:, :pred.shape[1]]
            x_t_prior, x_0 = self.noise_scheduler.estimate_x_t_prior_from_x_T(
                x_t, t, pred, clip_x0 = self.clip_x0, var_scale = pred_var_scale, cold_diffusion = cold_diffusion
            )
            x_T = pred 
            self_cond = x_0 
        else:
            raise ValueError("Unknown Objective")
        
        return x_t_prior, x_0, x_T, self_cond 


    @torch.no_grad()
    def denoise(self, x_t, lr_sample, steps=None, condition=None, use_ddim=False, **kwargs):
        self_cond = None 

        # ---------- run denoise loop ---------------
        steps = self.noise_scheduler.timesteps if steps is None else steps
        if use_ddim:
            # [0, 1, 2, ..., T-1] if steps = T 
            timesteps_array = torch.linspace(0, self.noise_scheduler.T-1, steps, dtype=torch.long, device=x_t.device) 
        else:
            # [0, ...,T-1] (target time not time of x_t)
            timesteps_array = self.noise_scheduler.timesteps_array[slice(0, steps)]
            
        for i, t in tqdm(enumerate(reversed(timesteps_array))):
            # UNet prediction
            x_t = torch.cat([x_t, lr_sample], dim=1) # condition is the second half of the tensor
            x_t, x_0, x_T, self_cond = self.forward(x_t, t.expand(x_t.shape[0]), condition, self_cond=self_cond, **kwargs)
            self_cond = self_cond if self.use_self_conditioning else None  
        
            if use_ddim and (steps - i - 1 > 0):
                t_next = timesteps_array[steps - i - 2]
                alpha = self.noise_scheduler.alphas_cumprod[t]
                alpha_next = self.noise_scheduler.alphas_cumprod[t_next]
                sigma = kwargs.get('eta', 1) * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
                c = (1 - alpha_next - sigma ** 2).sqrt()
                noise = torch.randn_like(x_t)
                x_t = x_0 * alpha_next.sqrt() + c * x_T + sigma * noise
        
        return x_t # Should be x_0 in final step (t=0)

    @torch.no_grad()
    def sample(self, lr_sample, condition=None, steps=None, use_ddim=False, **kwargs):
        template = torch.zeros((
            lr_sample.shape[0], 
            self.noise_estimator.in_channels - lr_sample.shape[1], 
            *self.noise_estimator.sample_size
            ), device = self.device
        )

        x_T = self.noise_scheduler.x_final(template)

        # concatenating with condition is done inside the denoising function
        x_0 = self.denoise(
            x_T,
            lr_sample, 
            steps = steps, 
            condition = condition.to(self.device, dtype=self.dtype) if condition is not None else None, 
            use_ddim = use_ddim,
            **kwargs
        ) # => [B, estimator.in_channels, W, H] (without condition)

        return x_0 

    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.ema_model.step(self.noise_estimator)
    
    def configure_optimizers(self):
        optimizer = self.optimizer(self.noise_estimator.parameters(), **self.optimizer_kwargs)
        if self.lr_scheduler is not None:
            lr_scheduler = {
                'scheduler': self.lr_scheduler(optimizer, **self.lr_scheduler_kwargs),
                'interval': 'step',
                'frequency': 1
            }
            return [optimizer], [lr_scheduler]
        else:
            return [optimizer]
        