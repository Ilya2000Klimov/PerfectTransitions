#!/bin/bash
#SBATCH -A cs175_class_gpu ## Account to charge
#SBATCH --time=16:00:00 ## Maximum running time of program
#SBATCH --nodes=1 ## Number of nodes.
## Set to 1 if you are using GPU.
#SBATCH --partition=free-gpu ## Partition name
#SBATCH --mem=64GB ## Allocated Memory
#SBATCH --cpus-per-task=16 ## Number of CPU cores
#SBATCH --gres=gpu:V100:1 ## Type and the number of GPUs
#SBATCH --output=SLURMlogs/feature_extract_%j.out
#SBATCH --error=SLURMlogs/feature_extract_%j.err
#SBATCH --mail-type=END,FAIL          # get emails for end/fail
#SBATCH --mail-user=Ilya2000Klimov@gmail.com       # your email address
## Don't change the GPU numbers.


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

python song_segmentation.py \
    --input_dir /path/to/full_songs \
    --output_dir /path/to/segmented_clips \
    --segment_length 25 \
    --overlap 5