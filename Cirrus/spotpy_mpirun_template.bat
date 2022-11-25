#!/bin/bash

#SBATCH --job-name="jobname"
#SBATCH --qos=short
#SBATCH --nodes=node_count
#SBATCH --ntasks-per-node=core_count
#SBATCH --chdir=working_dir
#SBATCH --account=misc
#SBATCH --error=logs/error_log
#SBATCH --partition=PARTITION
#SBATCH --output=logs/out_log
##SBATCH --mail-type=ALL

echo "Job uses a total of $SLURM_NTASKS cores"

module load anaconda
mpirun -n  $SLURM_NTASKS /home/susterph/env/bin/python -u mspot_script
