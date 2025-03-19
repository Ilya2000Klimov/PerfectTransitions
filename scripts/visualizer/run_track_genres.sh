#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --time=20:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=standard  ## Partition name
#SBATCH --mem=8GB            ## Allocated Memory
#SBATCH --cpus-per-task 8     ## Number of CPU cores
#SBATCH --output=simple_%j.out
#SBATCH --error=simple_%j.err
#SBATCH --mail-type=END,FAIL          # get emails for end/fail
#SBATCH --mail-user=iklimov@uci.edu       # your email address

srun python build_track_genres.py \
    --tracks_csv ../data/fma_metadata/tracks.csv \
    --genres_csv ../data/fma_metadata/genres.csv \
    --output_csv ../data/fma_metadata/track_genres.csv