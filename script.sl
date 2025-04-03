#!/bin/bash

# Slurm submission script, serial job
# CRIHAN v 1.00 - Jan 2017
# support@criann.fr

# Time limit for the calculation (48:00:00 max)
#SBATCH --time 72:00:00

# Memory to use (here 50Go)
#SBATCH --mem-per-gpu 50000

# Type of gpu to use, either gpu_all, gpu_k80, gpu_p100 or gpu_v100
#SBATCH --partition gpu_all

# Number of gpu to use
#SBATCH --gres gpu:8

# Number of tasks
#SBATCH --ntasks-per-node=8

# Number of node to use
#SBATCH --nodes 1

# Number of cpu to use
#SBATCH --cpus-per-task=6

# Were to write the logs
#SBATCH --error slurm/%J.err
#SBATCH --output slurm/%J.out

module load aidl/pytorch/2.0.0-cuda11.7
export PYTHONUSERBASE=/home/2021012/sruan01/riles/env
pip install pytorch-lightning==2.0.6 --user
pip install nibabel --user
pip install wandb --user
pip install omegaconf --user
pip install shortuuid --user
pip install monai --user
pip install einops --user
pip install lpips --user
pip install pytorch_msssim --user

# Start the calculation (safer to use srun)
srun python3 $1