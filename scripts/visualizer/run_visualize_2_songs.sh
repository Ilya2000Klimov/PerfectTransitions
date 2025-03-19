#!/bin/bash
#SBATCH --job-name=visualize_unified_embeddings
#SBATCH --partition=free-gpu        # Change to the correct GPU partition on your cluster
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=8           # Adjust CPU allocation (depends on data loading)
#SBATCH --mem=32G                   # 32GB system RAM
#SBATCH --time=12:00:00             # Set max training time
#SBATCH --output=SLURMlogs/visualize_unified_embeddings_%j.out
#SBATCH --error=SLURMlogs/visualize_unified_embeddings_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=iklimov@uci.edu

srun python visualize_unified_embeddings.py \
    --embeddings_dir ./../data/embeddings \
    --method pca \
    --dim 2 \
    --songA 005002 \
    --songB 029959 \
    --recursive \
    --perplexity 40
