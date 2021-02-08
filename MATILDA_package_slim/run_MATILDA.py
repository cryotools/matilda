# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## import of necessary packages
import pandas as pd

## Setting file paths and parameters
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package/"
input_path_data = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/"
input_path_observations = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"

data_csv = "no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
observation_data = "runoff_bashkaindy_04_2019-11_2020_test.csv" # Daily Runoff Observations in mm

df = pd.read_csv(input_path_data + data_csv)
obs = pd.read_csv(input_path_observations + observation_data)

parameter = pd.Series({"cal_period_start": '2018-01-01 00:00:00', "cal_period_end":'2019-12-31 23:00:00',
                       "sim_period_start": '2019-01-01 00:00:00', "sim_period_end": '2020-11-01 23:00:00',
                       "plot_frequency": "D", "plot_frequency_long": "Daily",
                       "catchment_area":46.232, "glacier_area":2.566,
                       "elevation_data":3864, "elevation_glacier": 4042, "elevation_catchment":3360,
                       "lapse_rate_temperature":-0.006, "lapse_rate_precipitation":0,
                       "TT_snow":0, "TT_rain":2, "CFMAX_snow":2.8, "CFMAX_ice":5.6, "CFR_snow":0.05, "CFR_ice":0.05,
                       "BETA" : 1.0, "CET" : 0.15, "FC" : 250, "K0" : 0.055, "K1" : 0.055, "K2" : 0.04, "LP" : 0.7,
                        "MAXBAS" : 3.0, "PERC" : 1.5, "UZL" : 120, "PCORR" : 1.0, "SFCF" : 0.7, "CWH" : 0.1})

## Running MATILDA
df, obs = data_preproc(df, obs, parameter) # Data preprocessing

output_MATILDA = MATILDA(df, obs, parameter) # MATILDA model run + downscaling

fig1 = plots.plot_meteo(output_MATILDA[0], plot_frequency_long)

fig1, fig2, fig3 = plot_MATILDA(output_MATILDA[0], parameter)


fig2.show()