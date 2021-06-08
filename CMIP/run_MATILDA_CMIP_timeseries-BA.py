## Running all the required functions
from pathlib import Path; home = str(Path.home())
from datetime import datetime
import os
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA_slim import MATILDA


## Model configuration
# Directories
cmip_data = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/CMIP6/"
glacier_profile = pd.read_csv("/home/ana/Seafile/Masterarbeit/Data/glacier_profile.txt")
output_path = home + "/Seafile/Ana-Lena_Phillip/data/input_output/output/bash_kaindy_cmip4_5-2021-2100"

cmip_4_5 = pd.read_csv(cmip_data + "no182_CMIP6_ssp2_4_5_mean_2000_2100_41-75.9_fitted2ERA5Lfit.csv")
cmip_4_5.columns = ['TIMESTAMP', 'T2', 'RRR']

parameter = MATILDA.MATILDA_parameter(cmip_4_5, set_up_start='2015-01-01 12:00:00', set_up_end='2020-12-31 12:00:00',
                                          sim_start='2021-01-01 12:00:00', sim_end='2100-12-31 12:00:00', freq="Y",
                                          lat=41, area_cat=46.23, area_glac=2.566, ele_dat=2250, ele_glac=4035, ele_cat=3485,
                                        CFMAX_ice=2.5, CFMAX_snow=5)
df_preproc = MATILDA.MATILDA_preproc(cmip_4_5, parameter)
output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, glacier_profile=glacier_profile)
output_MATILDA = MATILDA_plots(output_MATILDA, parameter)
MATILDA_save_output(output_MATILDA, parameter, output_path) # save regular MATILDA run

