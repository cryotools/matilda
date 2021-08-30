#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --output=/data/scratch/tappeana/Masterarbeit/logs/run_MATILDA_bashkaingdy.out
#SBATCH --error=/data/scratch/tappeana/Masterarbeit/logs/run_MATILDA_bashkaingdy.err
#SBATCH --chdir=/data/scratch/tappeana/Masterarbeit/
#SBATCH --qos=medium

echo $SLURM_CPUS_ON_NODE

date
module load anaconda
python3 /data/scratch/tappeana/Masterarbeit/run_MATILDA_bashkaingdy.py
date
