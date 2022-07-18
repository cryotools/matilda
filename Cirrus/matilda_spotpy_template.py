# -*- coding: UTF-8 -*-

## import of necessary packages
import os

import pandas as pd
from pathlib import Path
import sys
import numpy as np
import socket
import spotpy
import matplotlib.pyplot as plt
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
# sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/MATILDA')
sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools')
from Preprocessing.Preprocessing_functions import dmod_score, load_cmip, cmip2df
from Test_area.SPOTPY import mspot
from MATILDA_slim import MATILDA


## Settings

reps = 60
algorithm = 'demcz'
cores = 20
dbname='matilda_par_smpl_test'

output_path = None
set_up_start='1982-01-01 00:00:00'; set_up_end='1984-12-31 23:00:00'
sim_start='1985-01-01 00:00:00'; sim_end='1987-12-31 23:00:00' 
freq="D"
dbformat='csv'
save_sim = False


## Setting file paths and parameters
    # Paths
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
input_path = wd + "/input/kyzylsuu"


t2m_path = "/met/era5l/t2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_path = "/met/era5l/tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
runoff_obs = "/hyd/obs/Kyzylsuu_1982_2021_latest.csv"
cmip_path = '/met/cmip6/'

    # Calibration period
t2m = pd.read_csv(input_path + t2m_path)
tp = pd.read_csv(input_path + tp_path)
df = pd.concat([t2m, tp.tp], axis=1)
df.rename(columns={'time': 'TIMESTAMP', 't2m': 'T2','tp':'RRR'}, inplace=True)
obs = pd.read_csv(input_path + runoff_obs)

    # Scenarios
cmipT = load_cmip(input_path + cmip_path, 't2m_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_')
cmipP = load_cmip(input_path + cmip_path, 'tp_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_')
glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")


## Run SPOTPY:

best_summary = mspot.psample(df=df, obs=obs, rep=reps, output=output_path,
                            set_up_start=set_up_start, set_up_end=set_up_end,
                            sim_start=sim_start, sim_end=sim_end, freq=freq,
                            area_cat=315.694, area_glac=32.51, lat=42.33, parallel=True, algorithm=algorithm, cores=cores,
                            dbname=dbname, dbformat=dbformat, save_sim=save_sim)




