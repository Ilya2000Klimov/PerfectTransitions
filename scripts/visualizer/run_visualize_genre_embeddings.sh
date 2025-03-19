#!/bin/bash
#SBATCH --job-name=visualize_unified_embeddings
#SBATCH --partition=free-gpu        # Change to the correct GPU partition on your cluster
#SBATCH --gres=gpu:1                # Request 1 GPU
#SBATCH --cpus-per-task=8           # Adjust CPU allocation (depends on data loading)
#SBATCH --mem=32G                   # 32GB system RAM
#SBATCH --time=12:00:00             # Set max training time
#SBATCH --output=SLURMlogs/visualize_unified_embeddings_%j.out
#SBATCH --error=SLURMlogs/visualize_unified_embeddings%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=iklimov@uci.edu

srun python visualize_embeddings_genre.py \
    --embeddings_dir ./../data/embeddings \
    --recursive \
    --method pca \
    --dim 2 \
    --perplexity 40 \
    --track_genres_csv ../data/fma_metadata/track_genres.csv \
    --genres_csv ../data/fma_metadata/genres.csv


