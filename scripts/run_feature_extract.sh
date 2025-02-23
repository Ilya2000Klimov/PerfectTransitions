#!/bin/bash
#SBATCH --job-name=beats_feature_extract
#SBATCH --partition=free-gpu           # or the correct GPU partition on your cluster
#SBATCH --gres=gpu:1                   # request 1 GPU
#SBATCH --cpus-per-task=8             # up to 8-16 cores as needed
#SBATCH --mem=32G                     # 32GB system RAM
#SBATCH --time=4:00:00                # 4 hours max
#SBATCH --output=SLURMlogs/feature_extract_%j.out
#SBATCH --error=SLURMlogs/feature_extract_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=Ilya2000Klimov@gmail.com

# --- Load Modules/Environment ---
module load cuda/12.1
module load python/3.10

# Activate your conda environment
source conda activate PT

echo "[Info] Starting BEATs feature extraction on $HOSTNAME"

# --- Run Python Script ---
srun python feature_extraction.py \
    --data_dir "./../data/segments" \
    --output_dir "./../data/embeddings" \
    --model_checkpoint "./../model_checkpoints/BEATs_iter3_plus_AS2M.pt" \
    --save_freq 500     --resume_if_checkpoint_exists True

echo "[Info] Feature extraction job completed."
