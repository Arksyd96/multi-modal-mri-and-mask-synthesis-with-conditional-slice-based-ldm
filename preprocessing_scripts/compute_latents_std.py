import argparse
import torch
from tqdm import tqdm
import sys

from modules.models.autoencoder import AutoencoderKL, VAE
from modules.data_preprocessing import BRATSDataModule

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--model-path', type=str, required=True)
    parser.add_argument('-d', '--data-path', type=str, required=True, help='Path to the data .npy file')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    autoencoder = VAE.load_from_checkpoint(args.model_path).to(device)
    autoencoder.eval()

    datamodule = BRATSDataModule(
        data_dir        = args.data_path,
        train_ratio     = 0.99, 
        batch_size      = 64,
        num_workers     = 32,
        dtype           = torch.float32,
        include_radiomics = False
    )

    datamodule.setup()

    std = 0
    with torch.no_grad():
        for idx, batch in enumerate(tqdm(datamodule.train_dataloader(), position=0, leave=True, desc='Encoding')):
            targets, positions = batch
            targets, positions = targets.to(device, dtype=torch.float32), positions.to(device, dtype=torch.long)
            positions = positions.squeeze(1)

            temb = None
            if autoencoder.positional_embedder is not None:
                if positions is None:
                    raise ValueError("Positional encoding requires position tensor. ``None`` was provided.")
                temb = autoencoder.encode_position(positions)

            z = autoencoder.encode(targets, temb)
            std = std + z.std()
        
        std = std / (idx + 1)

    print('standard deviation: {}'.format(std))
