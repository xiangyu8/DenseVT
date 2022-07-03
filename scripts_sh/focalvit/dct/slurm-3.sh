#!/bin/bash
#SBATCH --gres="gpu:4" 
#SBATCH  --job-name=7
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-8:00
sh ./scripts_sh/focalvit/dct/train_baseline_dct_attn_svhn.sh
sh ./scripts_sh/focalvit/dct/train_baseline_dct_attn_flowers.sh
