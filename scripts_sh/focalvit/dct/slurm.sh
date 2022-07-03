#!/bin/bash
#SBATCH --gres="gpu:4" 
#SBATCH  --job-name=8
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=0-12:30
sh ./scripts_sh/focalvit/dct/train_baseline_dct_attn.sh
sh ./scripts_sh/focalvit/dct/train_baseline_dct_attn_cifar10.sh
sh ./scripts_sh/focalvit/dct/train_baseline_dct_attn_cifar100.sh
