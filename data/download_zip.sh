#!/bin/bash
#SBATCH -A cs175_class        ## Account to charge
#SBATCH --time=20:00:00       ## Maximum running time of program
                              ## Set to 1 if you are using GPU.
#SBATCH --partition=standard  ## Partition name
#SBATCH --mem=16GB            ## Allocated Memory
#SBATCH --cpus-per-task 16     ## Number of CPU cores
#SBATCH --output=simple_%j.out
#SBATCH --error=simple_%j.err
#SBATCH --mail-type=END,FAIL          # get emails for end/fail
#SBATCH --mail-user=Ilya2000Klimov@gmail.com       # your email address

srun aria2c -x 16 -s 16 -c "https://os.unil.cloud.switch.ch/fma/fma_medium.zip"
