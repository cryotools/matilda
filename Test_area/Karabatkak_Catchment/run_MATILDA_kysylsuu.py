# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
import numpy as np
import socket
import matplotlib.pyplot as plt
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
# sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/MATILDA')
# sys.path.append(home + '/Ana-Lena_Phillip/data/scripts/Test_area')
sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/Preprocessing')
from Preprocessing_functions import dmod_score, load_cmip, cmip2df
from MATILDA_slim import MATILDA

## Setting file paths and parameters
    # Paths
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
input_path = wd + "/input/kyzylsuu"
output_path = wd + "/output/kyzylsuu"

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

    # Senarios
cmipT = load_cmip(input_path + cmip_path, 't2m_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_')
cmipP = load_cmip(input_path + cmip_path, 'tp_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_')
glacier_profile = pd.read_csv(wd + "/kyzulsuu_glacier_profile.csv")


# Basic overview plot
# obs_fig = obs.copy()
# obs_fig.set_index('Date', inplace=True)
# obs_fig.index = pd.to_datetime(obs_fig.index)
# # obs_fig = obs_fig[slice('1984-10-01','1985-01-31')]
# plt.figure()
# ax = obs_fig.plot(label='Kyzylsuu (Hydromet)')
# ax.set_ylabel('Discharge [m³/s]')
#
# plt.show()

##
output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs,  output=output_path, set_up_start='1982-01-01 00:00:00', set_up_end='1984-12-31 23:00:00',
                                      sim_start='1985-01-01 00:00:00', sim_end='1989-12-31 23:00:00', freq="D",
                                      area_cat=315.694, area_glac=32.51, lat=42.33, warn=True, # soi=[5, 10],
                                      ele_dat=2550, ele_glac=4074, ele_cat=3225, lr_temp=-0.0059, lr_prec=0,
                                      TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4, CFMAX_ice=6, CFR_snow=0.08765,
                                      CFR_ice=0.01132, BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
                                      LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765, AG=1000)

output_MATILDA[6].show()

## Mit default parameters:

# output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs,  output=output_path, set_up_start='1982-01-01 00:00:00', set_up_end='1984-12-31 23:00:00',
#                                       sim_start='1985-01-01 00:00:00', sim_end='1989-12-31 23:00:00', freq="D",
#                                       area_cat=315.694, area_glac=32.51, lat=42.33,# soi=[5, 10],
#                                       ele_dat=2550, ele_glac=4074, ele_cat=3225)
#
# output_MATILDA[6].show()

# - Letzten zwei Darstellungen zeigen mit und ohne glacier-to-soil option
# - mit der option macht gestackte darstellung eigentlich keinen sinn mehr
# - noch einmal mit "optimiertem" parametersatz probieren ob matilda wieder vollständig ohne die option und dann auf master pushen
# - danach option wieder einbauen und in separaten branch
# - eigentlich müsste man beides mal mit SPOTPY optimieren und dann betrachten




# test = df.set_index('TIMESTAMP')
# test['RRR'][slice('1982-01-01 00:00:00','1990-12-31 23:00:00')]
# for i in range(2100,2192):
#     print(output_MATILDA[0]['Q_DDM_scaled'].squeeze()[i])
# - glacier melt am Besten bei tosoil mit einfügen

## With glacier change

# df_scen = cmip2df(cmipT, cmipP, 'ssp1', 'mean')
#
# output_MATILDA = MATILDA.MATILDA_simulation(df_scen, output=None, set_up_start='2017-01-01 00:00:00', set_up_end='2020-12-31 23:00:00',
#                                       sim_start='2021-01-01 00:00:00', sim_end='2030-12-31 23:00:00', freq="D",
#                                       area_cat=315.694, area_glac=32.51, lat=42.33, glacier_profile= glacier_profile,
#                                       ele_dat=2550, ele_glac=4074, ele_cat=3225, lr_temp=-0.0059, lr_prec=-0.0002503,
#                                       TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4, CFMAX_ice=6, CFR_snow=0.08765,
#                                       CFR_ice=0.01132, BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
#                                       LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765)
# output_MATILDA[6].show()
#
# glac_area = output_MATILDA[4].iloc[:,:-1]
# glac_area = glac_area.set_index('time')
# glac_area.plot()
# plt.show()
## Validation

# Adapt when parametrization is set up:

# dmod_score(bc_check['sdm'], bc_check['y_predict'], bc_check['x_predict'], ylabel="Temperature [C]")



## Running MATILDA
# parameter = MATILDA.MATILDA_parameter(df, set_up_start='1987-01-01 00:00:00', set_up_end='1988-12-31 23:00:00',
#                                       sim_start='1992-01-01 00:00:00', sim_end='1995-07-30 23:00:00', freq="D",
#                                       area_cat=315.694, area_glac=32.51, lat=42.33,
#                                       ele_dat=2550, ele_glac=4074, ele_cat=3225, lr_temp=-0.005936, lr_prec=-0.0002503,
#                                       TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4.824, CFMAX_ice=5.574, CFR_snow=0.08765,
#                                       CFR_ice=0.01132, BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
#                                       LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765)
#
# df_preproc, obs_preproc = MATILDA.MATILDA_preproc(df, parameter, obs=obs)
# #
# output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, obs_preproc)
# #
# output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
# output_MATILDA[6].show()


# MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path)