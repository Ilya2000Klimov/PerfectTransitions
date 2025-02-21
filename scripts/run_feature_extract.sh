#!/bin/bash
#SBATCH --job-name=beats_feature_extract
#SBATCH --partition=free-gpu           # or the correct GPU partition on your cluster
#SBATCH --gres=gpu:1                   # request 1 GPU
#SBATCH --cpus-per-task=8             # up to 8-16 cores as needed
#SBATCH --mem=32G                     # 32GB system RAM
#SBATCH --time=4:00:00                # 4 hours max
#SBATCH --output=SLURMlogs/feature_extract_%j.out
#SBATCH --error=SLURMlogs/feature_extract_%j.err
#SBATCH --mail-type=END,FAIL          # get emails for end/fail
#SBATCH --mail-user=<YOUR_EMAIL>       # your email address

# --- Load Modules/Environment ---
# module purge
module load cuda/12.2  # example, match your cluster's CUDA version
module load python/3.11 # or whichever Python module is available

# If you use conda or virtualenv
source conda activate rl  # Or: conda activate myenv

# Optional: If your site has a special approach for preemption, 
# load checkpoint-resume modules or set environment variables.

echo "Starting feature extraction job on $HOSTNAME"

# --- Run Python Script ---
# If you have checkpoint logic in feature_extraction.py, pass --resume if needed.
srun python scripts/feature_extraction.py \
    --data_dir /data/dataset/fma/segments \
    --output_dir /data/dataset/embeddings \
    --model_checkpoint /model_checkpoints/BEATs_iter3_plus_AS2M.pt \
    --batch_size 1 \
    --save_freq 500 \
    --resume_if_checkpoint_exists True

echo "Feature extraction job completed."
