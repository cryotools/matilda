# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## import of necessary packages
import pandas as pd
from pathlib import Path

home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package_slim')
from MATILDA_slim import MATILDA

## Setting file paths and parameters
working_directory = home + "/Seafile/Ana-Lena_Phillip/data/"
input_path = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/"

data_csv = "obs_20210313_kyzylsuu_awsq_1982_2019.csv"  # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
runoff_obs = "obs_kashkator_runoff_2017_2018.csv"  # Daily Runoff Observations in m³/s
output_path = working_directory + "input_output/output/" + data_csv[4:21]

df = pd.read_csv(input_path + data_csv)
obs = pd.read_csv(input_path + runoff_obs)
# obs["Qobs"] = obs["Qobs"] / 86400*(46.232*1000000)/1000

## Parametrization


## Running MATILDA
parameter = MATILDA.MATILDA_parameter(df, set_up_start='2017-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                                      sim_start='2017-01-01 00:00:00', sim_end='2018-12-31 23:00:00', freq="D",
                                      area_cat=7.53, area_glac=2.95,
                                      ele_dat=2550, ele_glac=3957, ele_cat=3830)
df_preproc, obs_preproc = MATILDA.MATILDA_preproc(df, parameter, obs=obs)  # Data preprocessing

output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, obs_preproc)  # MATILDA model run + downscaling

output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
# Creating plot for the input (meteorological) data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3) and
# adding them to the output

# MATILDA_save_output(output_MATILDA, parameter, output_path)

## This function is a standalone function to run the whole MATILDA simulation
# If output = output_path in function, the output will be saved to a new folder
output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs, set_up_start='2018-01-01 00:00:00',
                                            set_up_end='2018-12-31 23:00:00',
                                            sim_start='2019-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D",
                                            area_cat=7.53, area_glac=2.95,
                                            ele_dat=2550, ele_glac=3957, ele_cat=3830, TT_snow=0, TT_rain=2)
output_MATILDA[7].show()

output_MATILDA[0].Q_Total

##
