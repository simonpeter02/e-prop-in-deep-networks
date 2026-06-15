#!/bin/bash
#SBATCH --job-name=eprop-exp5
#SBATCH --output=results/exp5_%j.log
#SBATCH --error=results/exp5_%j.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1

# Activate your conda/venv environment — adjust the path if needed
source ~/miniconda3/bin/activate base

python -m experiments.exp5_shd_alif
