import numpy as np
import argparse
from tqdm import tqdm

import torch
import torch.nn.functional as F

from modules.diffusion import DiffusionPipeline, TumorEnhancingDiffusionPipeline
from modules.models.autoencoder import AutoencoderKL, ConditionMLP, VAE
from modules.scheduler import GaussianNoiseScheduler
from modules.models.unet import UNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def deposit_tumor_patch(tumor_patch, binary_mask, original_sequence):
    """
    Deposit a tumor patch onto a full image based on the binary mask.

    Args:
    - tumor_patch (torch.Tensor): Tensor representing the tumor patch of shape (batch_size, channels, 64, 64).
    - binary_mask (torch.Tensor): Binary mask of the tumor in the full image, shape (batch_size, 1, H, W).
    - full_image_shape (tuple): Shape of the full image (H, W).

    Returns:
    - full_image_with_tumor (torch.Tensor): Full image with the tumor patch deposited at the appropriate location.
    """

    batch_size = tumor_patch.shape[0]
    complementary_mask = 1 - binary_mask
    tumor_placeholder = torch.ones(
        tumor_patch.shape[0],
        tumor_patch.shape[1],
        original_sequence.shape[2],
        original_sequence.shape[3]
    ).to(tumor_patch.device, dtype=tumor_patch.dtype) * -1 # empty placeholder

    for i in range(batch_size):
        # Find the coordinates of the tumor in the binary mask
        nonzero_indices = torch.nonzero(binary_mask[i].squeeze(0))
        min_x = nonzero_indices[:, 0].min()
        max_x = nonzero_indices[:, 0].max()
        min_y = nonzero_indices[:, 1].min()
        max_y = nonzero_indices[:, 1].max()

        # Calculate center coordinates
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2

        # Calculate offsets to place the tumor patch in the full image
        offset_x = max(0, center_x - 32)  # 32 = 64 // 2
        offset_y = max(0, center_y - 32)

        # Place the tumor patch in the full image
        tumor_placeholder[i, :, offset_x:offset_x + 64, offset_y:offset_y + 64] = tumor_patch[i]
        refined_sequence = tumor_placeholder[:, :-1, ...] * binary_mask + original_sequence * complementary_mask
        
        multilabel_mask = tumor_placeholder[:, -1, None]
        refined_sequence = torch.cat([refined_sequence, multilabel_mask], dim=1)

    return refined_sequence

def load_models(args):
    # loading autoencoder
    autoencoder : AutoencoderKL = AutoencoderKL.load_from_checkpoint(args.autoencoder_path).to(device)
    
    scheduler = GaussianNoiseScheduler(
        timesteps = 1000,
        schedule_strategy = 'scaled_linear'
    )

    condition_embedder = ConditionMLP(in_features=args.condition_features, out_features=128, hidden_dim=512)
    denoiser = UNet(
        sample_size = args.noise_shape[1:],
        in_ch = args.noise_shape[0],
        out_ch = args.noise_shape[0],
        spatial_dims = 3,
        hid_chs = [64, 128, 256, 512],
        strides = [1, 2, 2, 2],
        num_res_blocks = 2,
        kernel_sizes = [3, 3, 3, 3],
        temb_channels = 128,
        max_period = 1000,
        cond_embedder = condition_embedder,
        deep_supervision = False,
        use_res_block = True,
        use_attention = 'none'
    ).to(device)

    diffuser = DiffusionPipeline.load_from_checkpoint(
        args.diffuser_path,
        latent_embedder = autoencoder,
        noise_scheduler = scheduler,
        noise_estimator = denoiser,
        std_norm = args.std_norm, # 0.9760941863059998
        loss = torch.nn.L1Loss,
        clip_x0 = False,
        slice_based = True
    ).to(device)

    tumor_denoiser = UNet(
        sample_size = (64, 64),
        in_ch = 5,
        out_ch = 3,
        spatial_dims = 2,
        hid_chs = [64, 128, 256, 512],
        strides = [1, 2, 2, 2],
        kernel_sizes = [3, 3, 3, 3],
        temb_channels = 128,
        max_period = 1000,
        cond_embedder = None,
        deep_supervision = False,
        use_res_block = True,
        use_attention = 'none'
    ).to(device)

    tumor_enhancer = TumorEnhancingDiffusionPipeline.load_from_checkpoint(
        args.tumor_enhancer_path,
        noise_estimator = tumor_denoiser, 
        noise_scheduler = scheduler, 
        latent_embedder = autoencoder
    ).to(device)

    return autoencoder, diffuser, tumor_enhancer


def main(args):
    print(f'Using device: {device}')
    torch.cuda.empty_cache()
    
    autoencoder, diffuser, tumor_enhancer = load_models(args)
    autoencoder.eval()
    diffuser.eval()
    tumor_enhancer.eval()
    
    samples, refined_samples = [], []

    args.batch_size = min(args.batch_size, args.num_samples)
    for _ in tqdm(range(0, args.num_samples, args.batch_size), position=0, leave=True):
        if args.condition is None:
            x, y, z = F.one_hot(torch.randint(0, 2, size=(args.batch_size, 3)), num_classes = 2).unbind(dim=1)
            voxel_volume, sphericity = F.one_hot(torch.randint(1, 3, size=(args.batch_size, 2)), num_classes = 3).unbind(dim=1)
            condition = torch.cat([x, y, z, voxel_volume, sphericity], dim=1)
        else:
            condition = args.condition
        condition = condition.to(device)
        
        latents = diffuser.sample(
            num_samples = args.batch_size, 
            img_size = (diffuser.noise_estimator.in_channels, *diffuser.noise_estimator.sample_size),
            condition = condition,
            steps = args.steps,
            use_ddim = args.use_ddim
        )

        volumes = diffuser.decode_latents(latents.detach(), channels_first=True)
        volumes = volumes.clamp(-1, 1)
        volumes[:, -1] = (volumes[:, -1, ...] > 0).type(torch.float32)
        samples.append(volumes.detach().cpu().numpy())

        # refine tumors
        for p_idx in range(volumes.shape[0]):
            binary_mask = volumes[p_idx, -1, None, ...].permute(3, 0, 1, 2)

            tumoral_slices_map = binary_mask.sum(dim=(1, 2, 3)) > 20 # threshold of 20 pixels
            binary_mask = binary_mask[tumoral_slices_map]

            if binary_mask.__len__() > 0: # if mask not empty, otherwise skip
                lr_images = volumes[p_idx, :-1, ...].permute(3, 0, 1, 2)
                lr_images = lr_images[tumoral_slices_map]

                lr_tumors = tumor_enhancer._extract_tumor_patch(lr_images, binary_mask)
                hr_tumors = tumor_enhancer.sample(lr_tumors, steps = 300, use_ddim = True)
                hr_tumors = hr_tumors.clamp(-1, 1)

                refined_slices = deposit_tumor_patch(hr_tumors, binary_mask, lr_images)
                refined_slices[:, -1] = refined_slices[:, -1].add(1).div(2).mul(3).round() # [-1, 1] => discrete classes

                volumes[p_idx, :, :, :, tumoral_slices_map] = refined_slices.permute(1, 2, 3, 0)

        refined_samples.append(volumes.detach().cpu().clone().numpy())

    samples = np.concatenate(samples, axis=0)
    refined_samples = np.concatenate(refined_samples, axis=0)

    # saving volume as .npy
    np.save(args.output, samples)
    np.save(args.output.replace('.npy', '_refined.npy'), refined_samples)

    print(f'Saved samples to {args.output}')
    print(f'Saved refined samples to {args.output.replace(".npy", "_refined.npy")}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Sampling from the model')
    parser.add_argument('-ap', '--autoencoder-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('-dp', '--diffuser-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('-tp', '--tumor-enhancer-path', type=str, required=True, help='Path to the model checkpoint')
    parser.add_argument('-n', '--num-samples', type=int, default=8, help='Number of samples to generate')
    parser.add_argument('-s', '--steps', type=int, default=1000, help='Number of steps to sample')
    parser.add_argument('-ddim', '--use-ddim', action='store_true', help='Use DDIM for sampling')
    parser.add_argument('-bs', '--batch-size', type=int, default=8, help='Batch size for sampling')
    parser.add_argument('-ns', '--noise-shape', type=int, nargs='+', default=[8, 16, 16, 64], 
                        help='Shape of the noise tensor, with channels specified at axis = 0')
    parser.add_argument('-sn', '--std-norm', type=float, default=1.0, help='Standard deviation for the noise tensor')
    parser.add_argument('-o', '--output', type=str, default='./samples.npy', help='Output file to save samples as .npy - Specify location')
    
    # condition specific arguments
    parser.add_argument('-cf', '--condition-features', type=int, default=12, help='Size of the condition vector')
    parser.add_argument('-c', '--condition', type=int, nargs='+', help='Condition to sample from')

    args = parser.parse_args()
    main(args)