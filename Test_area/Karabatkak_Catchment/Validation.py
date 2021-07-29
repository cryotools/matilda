## Running all the required functions
from pathlib import Path; home = str(Path.home())
from datetime import datetime
import os
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from MATILDA_slim import MATILDA

##
input_df = home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/era5l/kyzylsuu_ERA5_Land_1982_2020_42.2_78.2_fitted2AWS.csv"
glacier_profile = home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/karabatkak_glacier_profile_farinotti.csv"
df = pd.read_csv(input_df)
df.columns = ['TIMESTAMP', 'T2', 'RRR']
glacier_profile = pd.read_csv(glacier_profile)

##
parameter = MATILDA.MATILDA_parameter(df, set_up_start='2001-01-01 12:00:00',
                                      set_up_end='2001-12-31 12:00:00',
                                      sim_start='2002-01-01 12:00:00', sim_end='2017-08-31 12:00:00', freq="Y",
                                      lat=42.25,
                                      area_cat=7.527, area_glac=2.271, ele_dat=2550, ele_glac=3830, ele_cat=3830,
                                      lr_temp=-0.005936,
                                      lr_prec=-0.0002503, TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4.824,
                                      CFMAX_ice=5.574, CFR_snow=0.08765, CFR_ice=0.01132, BETA=2.03, CET=0.0471,
                                      FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277, LP=0.4917, MAXBAS=2.494, PERC=1.723,
                                      UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765)
df_preproc = MATILDA.MATILDA_preproc(df, parameter)
output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, glacier_profile=glacier_profile)
output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
MATILDA.MATILDA_save_output(output_MATILDA, parameter, home + "/Seafile/Ana-Lena_Phillip/data/input_output/output/new_deltaH/Kashkator")