#!/bin/sh -l
# FILENAME: base_galign.slurm
#SBATCH --account=rajivak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --qos=normal

set -euo pipefail

export TF_FORCE_GPU_ALLOW_GROWTH=true
export TF_GPU_ALLOCATOR=cuda_malloc_async

USER_NAME=${USER:-$(id -un)}
export DATASET_ROOT=/scratch/gilbreth/${USER_NAME}/datasets
export TFDS_DATA_DIR="${DATASET_ROOT}/tensorflow_datasets"

/bin/hostname

export WANDB_API_KEY='89430f73707e42a5c4b69b47e4139e1b13246223'
PROJECT_PREFIX='moLO'
RUN_GROUP="slurm_$(date +%Y%m%d-%H%M%S)"

export WANDB_PROJECT="${PROJECT_PREFIX}"
export WANDB_GROUP="${RUN_GROUP}"
export WANDB_NAME="${RUN_GROUP}"
export WANDB_RESUME="never"
export WANDB_TAGS="grad_align"

TRAIN_LOG_DIR="/scratch/gilbreth/${USER_NAME}/molo_grad_allign/${RUN_GROUP}"

python -m celo.train \
    --optimizer celo_phase1 \
    --train_log_dir="${TRAIN_LOG_DIR}/phase1" \
    --train_partial \
    --outer_iterations 100000 \
    --max_unroll_length 2000 \
    --seed 0 \
    --trainer pes \
    --aug reparam \
    --aug_reparam_level global \
    --name train_celo_phase1 \
    --outer_lr 3e-4 \
    --task fast_velo \
    --log_grad_alignment \
    --disable_wandb 