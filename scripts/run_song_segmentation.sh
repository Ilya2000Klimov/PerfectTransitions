#!/bin/bash
#SBATCH -A cs175_class_gpu ## Account to charge
#SBATCH --time=16:00:00 ## Maximum running time of program
#SBATCH --nodes=1 ## Number of nodes.
## Set to 1 if you are using GPU.
#SBATCH --partition=free-gpu ## Partition name
#SBATCH --mem=64GB ## Allocated Memory
#SBATCH --cpus-per-task=8 ## Number of CPU cores
#SBATCH --output=SLURMlogs/song_segment_%j.out
#SBATCH --error=SLURMlogs/song_segment_%j.err
#SBATCH --mail-type=END,FAIL          # get emails for end/fail
#SBATCH --mail-user=iklimov@uci.edu       # your email address
## Don't change the GPU numbers.


# --- Load Modules/Environment ---
# module purge
# module load cuda/12.1  # example, match your cluster's CUDA version
# module load python/3.10 # or whichever Python module is available

# If you use conda or virtualenv
source conda activate PT

# Optional: If your site has a special approach for preemption, 
# load checkpoint-resume modules or set environment variables.

echo "Starting song segmentation job on $HOSTNAME"

# --- Run Python Script ---
# If you have checkpoint logic in feature_extraction.py, pass --resume if needed.

srun python song_segmentation.py \
    --input_dir ./../data/fma_medium/ \
    --output_dir ./../data/segments \
    --segment_length 5 \
    --overlap 0