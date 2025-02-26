#!/bin/bash
#SBATCH --job-name=beats_lstm_sweep
#SBATCH --partition=free-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=SLURMlogs/lstm_sweep_%j.out
#SBATCH --error=SLURMlogs/lstm_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Ilya2000Klimov@gmail.com

# If using Conda environment:
source ~/.bashrc && conda activate PT                   # Activate your Conda environment

# Ensure WANDB API key is set (if needed for tracking)
export WANDB_API_KEY="2fb10bfa91d0f9638ec2a709e53c2ab05f843cd6"

echo "[INFO] Starting W&B Sweep on $HOSTNAME"

srun wandb agent search-byol/BEATs-LSTM-Transitions/xyz123 --count 100

echo "[INFO] Sweep Completed!"
