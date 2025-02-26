#!/bin/bash
#SBATCH --job-name=beats_lstm_train
#SBATCH --partition=free-gpu        # Change to the correct GPU partition on your cluster
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=8           # Adjust CPU allocation (depends on data loading)
#SBATCH --mem=32G                   # 32GB system RAM
#SBATCH --time=12:00:00             # Set max training time
#SBATCH --output=SLURMlogs/lstm_train_%j.out
#SBATCH --error=SLURMlogs/lstm_train_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>


# If using Conda environment:
source ~/.bashrc && conda activate PT                   # Activate your Conda environment

# Ensure WANDB API key is set (if needed for tracking)
export WANDB_API_KEY="2fb10bfa91d0f9638ec2a709e53c2ab05f843cd6"

# --- Start Training ---
echo "[INFO] Starting LSTM Training on $HOSTNAME"

srun python train_lstm.py \
    --embeddings_dir "./../data/embeddings" \
    --checkpoint_dir "./../model_checkpoints/lstm" \
    --resume_if_checkpoint_exists True

echo "[INFO] LSTM Training Completed!"
