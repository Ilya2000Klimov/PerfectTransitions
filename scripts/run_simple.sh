#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --time=20:00:00       ## Maximum running time of program
#SBATCH --nodes=1             ## Number of nodes.
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=standard  ## Partition name
#SBATCH --mem=2GB            ## Allocated Memory
#SBATCH --cpus-per-task 2     ## Number of CPU cores
#SBATCH --output=simple_%j.out
#SBATCH --error=simple_%j.err
#SBATCH --mail-type=END,FAIL          # get emails for end/fail
#SBATCH --mail-user=iklimov@uci.edu       # your email address

conda install --force-reinstall numpy scipy librosa torch torchaudio




