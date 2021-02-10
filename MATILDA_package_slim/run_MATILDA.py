# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## import of necessary packages
import pandas as pd
from MATILDA_slim import MATILDA_plots, MATILDA_simulation, MATILDA_preparation, MATILDA_submodules

## Setting file paths and parameters
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/"
input_path_data = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/"
input_path_observations = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"

data_csv = "no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
observation_data = "runoff_bashkaindy_04_2019-11_2020_test.csv" # Daily Runoff Observations in mm

output_path = working_directory + "input_output/output/" + data_csv[:15]


df = pd.read_csv(input_path_data + data_csv)
obs = pd.read_csv(input_path_observations + observation_data)

## Running MATILDA
parameter = MATILDA_preparation.MATILDA_parameter(df, set_up_start='2018-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                       sim_start='2019-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D", area_cat=46.232, area_glac=2.566,
                       ele_dat=3864, ele_glac=4042, ele_cat=3360, MAXBAS=9)
df, obs = MATILDA_preparation.MATILDA_preproc(df, obs, parameter) # Data preprocessing

output_MATILDA = MATILDA_submodules.MATILDA(df, obs, parameter) # MATILDA model run + downscaling

output_MATILDA = MATILDA_plots.MATILDA_plots(output_MATILDA, parameter)
# Creating plot for the input (meteorological) data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3) and
# adding them to the output

MATILDA_preparation.MATILDA_save_output(output_MATILDA, parameter, output_path)

# If output = output_path in function, the output will be saved to a new folder
output_MATILDA = MATILDA_simulation.MATILDA_simulation(df, obs, set_up_start='2018-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                       sim_start='2019-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D", area_cat=46.232, area_glac=2.566,
                       ele_dat=3864, ele_glac=4042, ele_cat=3360, MAXBAS=8)

