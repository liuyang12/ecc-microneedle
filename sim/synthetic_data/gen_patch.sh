#!/bin/bash

# Slurm sbatch options
#SBATCH -o log/gen_patch.log-%j
#SBATCH -c 48

# Initialize the module command first
source /etc/profile

# Load Anaconda and MPI module
module load anaconda/2021b

# Call your script as you would from the command line
let N=1000*$2
matlab -batch "GenerateDotsRandomPatch($1,$N)"