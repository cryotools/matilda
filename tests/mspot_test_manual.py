# -*- coding: UTF-8 -*-
import os
import pandas as pd
import argparse
from matilda.mspot_glacier import psample

# Argument parser setup
parser = argparse.ArgumentParser(description="Run the mspot_glacier sampling script with adjustable parameters.")
parser.add_argument('--glacier_only', type=bool, default=False, help='Set to True to enable glacier-only mode.')
parser.add_argument('--parallel', type=bool, default=False, help='Set to True to enable parallel processing.')
parser.add_argument('--cores', type=int, default=2, help='Number of cores to use for parallel processing.')
parser.add_argument('--rep', type=int, default=3, help='Number of repetitions for the simulation.')

args = parser.parse_args()

try:
    # Case 1: Script execution
    home = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Case 2: Interactive environment (e.g., Jupyter, IDE)
    cwd = os.getcwd()
    if os.path.basename(cwd) == 'tests':
        home = cwd
    elif os.path.basename(cwd) == 'matilda':
        home = os.path.join(cwd, 'tests')
    elif os.path.basename(cwd) == 'mspot_test_output':
        home = os.path.dirname(cwd)
    else:
        raise ValueError("Unknown working directory. Please specify the directory manually.")

print(f"The home directory is set to: {home}")

## Command-line arguments
glacier_only = args.glacier_only
parallel = args.parallel
cores = args.cores
rep = args.rep

## Input data
df = pd.read_csv(f'{home}/test_input/era5.csv')
obs = pd.read_csv(f'{home}/test_input/obs_runoff_example.csv')
swe = pd.read_csv(f'{home}/test_input/swe.csv')
glacier_profile = pd.read_csv(f'{home}/test_input/glacier_profile.csv')
output_path = f'{home}/mspot_test_output'

## Setup
settings = {
    'area_cat': 295.67484249904464,
    'area_glac': 31.829413146585885,
    'ele_cat': 3293.491688025922,
    'ele_dat': 3335.668840874115,
    'ele_glac': 4001.8798828125,
    'elev_rescaling': True,
    'freq': 'D',
    'lat': 42.18280043250193,
    'set_up_end': '1999-12-31',
    'set_up_start': '1998-01-01',
    'sim_end': '2020-12-31',
    'sim_start': '2000-01-01',
    'glacier_profile': glacier_profile,
    'rep': rep,
    'glacier_only': glacier_only,
    'obj_dir': "maximize",
    'parallel': parallel,
    'algorithm': 'lhs',
    'cores': cores,
    'dbname': 'mspot_lhs_test',
    'dbformat': 'csv',
    'save_sim': False,
    'target_mb': -430
}

## Some random settings
lim_dict = {'BETA_lo': 1.03, 'BETA_up': 2.67,
            'TT_snow_lo': -1.45, 'TT_snow_up': -0.61,
            'CFMAX_snow_lo': 4.35, 'CFMAX_snow_up': 6.11}

## Sampling runs
best_summary = psample(df, obs=obs, output=output_path, **settings,
                       fix_param=['SFCF', 'CET', 'FC', 'K0', 'K1', 'K2', 'MAXBAS', 'PERC', 'UZL', 'CWH', 'AG', 'LP',
                                  'CFR'],
                       fix_val={'SFCF': 1, 'CET': 0},
                       **lim_dict,
                       target_swe=swe,
                       swe_scaling=0.928
                       )
