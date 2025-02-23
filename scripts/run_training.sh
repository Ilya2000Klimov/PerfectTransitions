#!/bin/bash
#SBATCH --job-name=beats_lstm_train
#SBATCH --partition=free-gpu            # GPU partition
#SBATCH --gres=gpu:1                    # request 1 GPU
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                       # 64GB or more for training
#SBATCH --time=12:00:00                 # up to 12 hours
#SBATCH --output=SLURMlogs/train_lstm_%j.out
#SBATCH --error=SLURMlogs/train_lstm_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<YOUR_EMAIL>

# module purge
# module load cuda/11.7
# module load python/3.9

# Activate environment
source conda activate PT

echo "Starting LSTM training job on $HOSTNAME"

srun python scripts/train_lstm.py \
    --embeddings_dir /NFS/dataset/embeddings \
    --checkpoint_dir /NFS/model_checkpoints \
    --batch_size 32 \
    --margin 0.3 \
    --lstm_layers 2 \
    --hidden_dim 128 \
    --lr 0.001 \
    --max_epochs 50 \
    --patience 5 \
    --resume_if_checkpoint_exists True

echo "Training job completed."
