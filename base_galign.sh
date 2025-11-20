#!/bin/sh -l
# FILENAME: mo_celo_v_b_phase2.slurm
#SBATCH --account=rajivak
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=14
#SBATCH --gpus-per-node=1
#SBATCH --time=4:00:00
#SBATCH --partition=a100-80gb
#SBATCH --gres=gpu:1
#SBATCH --mem=100G
#SBATCH --qos=standby

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

python -m celo.train 
    --optimizer celo_phase1 \
    --train_partial \
    --outer_iterations 1 \
    --max_unroll_length 2000 \
    --seed 0 \
    --trainer pes \
    --aug reparam \
    --aug_reparam_level global \
    --name train_celo_phase1 \
    --outer_lr 3e-4 \
    --task fast_velo \
    --log_grad_alignment

python -m celo.celo.train \
    --optimizer=celo \
    --train_partial \
    --lstm_hidden_size=256 \
    --mlp_hidden_size=32 \
    --celo_param_inits=64 \
    --init_from_ckpt="${TRAIN_LOG_DIR}/phase1/checkpoints/train_celo_phase1/theta.state" \
    --outer_iterations=100000 \
    --max_unroll_length=2000 \
    --trainer=pes \
    --seed=0 \
    --name=train_celo_phase2 \
    --task=fast_velo \
    --outer_lr=3e-4 \
    --aug=reparam \
    --aug_reparam_level=global \
    --train_log_dir="${TRAIN_LOG_DIR}/phase2" \
    --ckpt_save_dir="${TRAIN_LOG_DIR}/phase2/checkpoints"
