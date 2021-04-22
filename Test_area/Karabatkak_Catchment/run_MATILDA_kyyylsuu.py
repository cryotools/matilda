# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package_slim')
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/Test_area')
import mspot
from MATILDA_slim import MATILDA
import matplotlib.pyplot as plt


## Setting file paths and parameters
working_directory = home + "/Seafile/Ana-Lena_Phillip/data/"
input_path = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/"

data_csv = "ERA5/20210313_42.25-78.25_kyzylsuu_awsq_1982_2019.csv"  # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
runoff_obs = "obs_kyzylsuu_runoff_1994_1997_zero.csv"  # Daily Runoff Observations in m³/s
#runoff_obs = "obs_kashkator_runoff_2017_2018.csv"
output_path = working_directory + "input_output/output/" + data_csv[4:21]

df = pd.read_csv(input_path + data_csv)
obs = pd.read_csv(input_path + runoff_obs)
# obs["Qobs"] = obs["Qobs"] / 86400*(46.232*1000000)/1000

## Parametrization

karab_par = mspot.psample(df=df, obs=obs, rep=3, set_up_start='2017-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
              sim_start='2017-01-01 00:00:00', sim_end='2018-11-01 23:00:00', freq="D", area_cat=315.69,
              area_glac=2.95, ele_dat=2550, ele_glac=3957, ele_cat=3221, lr_temp_lo=-0.0065, lr_temp_up=-0.005)

##
parameter = MATILDA.MATILDA_parameter(df, set_up_start='2017-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                                      sim_start='2017-01-01 00:00:00', sim_end='2018-12-31 23:00:00', freq="D",
                                      area_cat=315.69, area_glac=2.95,
                                      ele_dat=2550, ele_glac=3957, ele_cat=3221)
## Running MATILDA
parameter = MATILDA.MATILDA_parameter(df, set_up_start='1993-01-01 00:00:00', set_up_end='1993-12-31 23:00:00',
                                      sim_start='1994-01-01 00:00:00', sim_end='1997-12-31 23:00:00', freq="W",
                                      area_cat=315.69, area_glac=31.51,
                                      ele_dat=2550, ele_glac=4000, ele_cat=3221)
df_preproc, obs_preproc = MATILDA.MATILDA_preproc(df, parameter, obs=obs)  # Data preprocessing

output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, obs_preproc)  # MATILDA model run + downscaling

output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
# Creating plot for the input (meteorological) data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3) and
# adding them to the output
output_MATILDA[6].show()

output = plot_data(output_MATILDA, parameter)
test = plot_runoff(output, parameter)
test.show()

MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path)

## This function is a standalone function to run the whole MATILDA simulation
# If output = output_path in function, the output will be saved to a new folder
output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs, output=output_path, set_up_start='1993-01-01 00:00:00', set_up_end='1993-12-31 23:00:00',
                                      sim_start='1994-01-01 00:00:00', sim_end='1997-12-31 23:00:00', freq="D",
                                      area_cat=315.69, area_glac=31.51,
                                      ele_dat=2550, ele_glac=4000, ele_cat=3221, lr_temp=-0.006, lr_prec=0,
                                            BETA=3.6,CET=0.10,FC=500,K0=0.39, K1=0.16, K2=0.12,LP=0.53,
                                            MAXBAS=2,PERC=0.42,UZL=353,PCORR=1.1,TT_snow=-1.45,TT_rain=1.1,
                                            CFMAX_snow=3.14,CFMAX_ice=4.48,SFCF=0.58, CFR_snow=0.08,CFR_ice=0,
                                            CWH=0.14)
output_MATILDA[6].show()

output_MATILDA[0].Q_Total

##
