#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --output=/data/projects/ebaca/Ana-Lena_Phillip/data/matilda/Cirrus/logs/MATILDA_jyrgalang_sceua_10.out
#SBATCH --error=/data/projects/ebaca/Ana-Lena_Phillip/data/matilda/Cirrus/logs/MATILDA_jyrgalang_sceua_10.err
#SBATCH --chdir=/data/projects/ebaca/Ana-Lena_Phillip/data/matilda/Test_area/Jyrgalang_catchment/
#SBATCH --qos=medium

echo $SLURM_CPUS_ON_NODE

date
module load anaconda
python3 /data/projects/ebaca/Ana-Lena_Phillip/data/matilda/Test_area/Jyrgalang_catchment/spotpy_matilda_simple-JYRGALANG.py
date
