#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --output=/data/projects/ebaca/Ana-Lena_Phillip/data/scripts/Cirrus/logs/run_MATILDA_karabatkak.out
#SBATCH --error=/data/projects/ebaca/Ana-Lena_Phillip/data/scripts/Cirrus/logs/run_MATILDA_karabatkak.err
#SBATCH --chdir=/data/projects/ebaca/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/
#SBATCH --qos=medium

echo $SLURM_CPUS_ON_NODE

date
module load anaconda
python3 /data/projects/ebaca/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/run_MATILDA_karabatkak.py
date
