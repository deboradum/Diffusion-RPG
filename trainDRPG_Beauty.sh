#!/bin/bash

#SBATCH --job-name=DRPG_beauty
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=18
#SBATCH --time=08:30:00
#SBATCH --output=runs/drpg/beauty/run-%x-%j.out
#SBATCH --error=runs/drpg/beauty/run-%x-%j.err

set -euo pipefail

PROJECT_DIR="/projects/prjs2120/groups/group_16/code/RPG_KDD2025"

cd "${PROJECT_DIR}"

mkdir -p runs/drpg/beauty

module purge
module load 2025
module load Anaconda3/2025.06-1

~/.conda/envs/diffgm/bin/python -u main.py \
  --category=Beauty \
  --train_batch_size=1024 \
  --model=DRPG \
  --n_codebook=4 \
  --lr=0.01 \
  --n_layer=1 \
  --n_embd=256 \
  --num_beams=20 \
  --n_edges=200 \
  --n_views=4 \
  --embd_pdrop=0.3 \
  --attn_pdrop=0.3 \
  --diffusion_layers=4 \
  --diffusion_heads=4 \
  --sent_emb_model="sentence-transformers/sentence-t5-base" \
  --sent_emb_dim=768 \
  --sent_emb_pca=256 \
  --sent_emb_batch_size=256 \
  --denoise_inference_steps=4
