#!/bin/bash
#SBATCH -A cs175_class_gpu
#SBATCH --job-name=beats_lstm_sweep
#SBATCH --partition=gpu
#SBATCH --nodes=1             ## Number of nodes.
#SBATCH --gres=gpu:V100:1     ## Type and the number of GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=SLURMlogs/lstm_sweep_%j.out
#SBATCH --error=SLURMlogs/lstm_sweep_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Ilya2000Klimov@gmail.com

# If using Conda environment:
source ~/.bashrc && conda activate PT                   # Activate your Conda environment

# Ensure WANDB API key is set (if needed for tracking)
export WANDB_API_KEY="2fb10bfa91d0f9638ec2a709e53c2ab05f843cd6"

echo "[INFO] Starting W&B Sweep on $HOSTNAME"

srun wandb agent search-byol/PerfectTransitions-scripts/k6nozzjq --count 100

echo "[INFO] Sweep Completed!"
