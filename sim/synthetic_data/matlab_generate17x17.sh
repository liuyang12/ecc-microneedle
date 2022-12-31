#!/bin/bash

# Slurm sbatch options
#SBATCH -o log/matlab_generate17x17_1M.log-%j
#SBATCH -c 48

# Initialize the module command first
source /etc/profile

# Load Anaconda and MPI module
module load anaconda/2021b

# Call your script as you would from the command line
matlab -batch "GenerateDotsRandom17x17"