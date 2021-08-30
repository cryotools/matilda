# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## import of necessary packages
import pandas as pd
from pathlib import Path
from MATILDA_slim import MATILDA
import matplotlib.pyplot as plt
home = str(Path.home())

## Setting file paths and parameters
working_directory = home + "/Seafile/Ana-Lena_Phillip/data/"
input_path_data = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/Old/"
input_path_observations = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"

data_csv = "no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
observation_data = "runoff_bashkaindy_04_2019-11_2020_temp_limit.csv" # Daily Runoff Observations in mm

output_path = working_directory + "input_output/output/" + data_csv[:15]


df = pd.read_csv(input_path_data + data_csv)
obs = pd.read_csv(input_path_observations + observation_data)

#obs["Qobs"] = obs["Qobs"] / 86400*(46.232*1000000)/1000 # in der Datei sind die mm Daten, deswegen hier nochmal umgewandelt in m3/s
obs = obs.drop(columns=['Unnamed: 2', 'Unnamed: 3'])
#area_cat=46.232

## Running MATILDA
parameter = MATILDA.MATILDA_parameter(df, set_up_start='2018-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                       sim_start='2019-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D", lat=41, area_cat=46.232,
                                      area_glac=2.566, ele_dat=3864, ele_glac=4035, ele_cat=3485, CFMAX_ice=5,
                                      CFMAX_snow=2.5, BETA=1, CET=0.15, FC=200, K0=0.055, K1= 0.055, K2=0.04,LP=0.7,
                                      MAXBAS=2, PERC=2.5, UZL=60, TT_snow=-0.5, TT_rain=2, SFCF=0.7, CFR_ice=0.05,
                                      CFR_snow= 0.05, CWH=0.1)
df_preproc, obs_preproc = MATILDA.MATILDA_preproc(df, parameter, obs=obs_poly) # Data preprocessing

output_MATILDA = MATILDA_submodules(df_preproc, parameter, obs_preproc) # MATILDA model run + downscaling

output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
output_MATILDA[6].show()
# Creating plot for the input (meteorological) data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3) and
# adding them to the output

#MATILDA_save_output(output_MATILDA, parameter, output_path)

## This function is a standalone function to run the whole MATILDA simulation
# If output = output_path in function, the output will be saved to a new folder
output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs, set_up_start='2018-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                       sim_start='2019-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D", area_cat=46.232, area_glac=2.566,
                       ele_dat=3864, ele_glac=4035, ele_cat=3485, TT_snow=0, TT_rain=2)
output_MATILDA[4].show()

output_MATILDA[0].Q_Total
