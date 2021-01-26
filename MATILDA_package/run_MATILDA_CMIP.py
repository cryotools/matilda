## Running all the required functions
from datetime import datetime
import os
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from MATILDA import dataformatting # prepare and format the input and output data
from MATILDA import DDM # importing the DDM model functions
from MATILDA import HBV # importing the HBV model function
from MATILDA import stats, plots # importing functions for statistical analysis and plotting

## Model configuration
# Directories
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package/"
input_path_data = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/"
input_path_observations = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"

data_csv = "no182ERA5_Land_2000_2020_fitted.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
observation_data = "runoff_bashkaindy_04_2019-11_2020_test.csv" # Daily Runoff Observations in mm
cmip_data = "/home/ana/Seafile/Tianshan_data/CMIP/CMIP5/CMIP5_monthly_trend.csv"

# Additional information
# Time period for the spin up
cal_period_start = '2001-01-01 00:00:00' # beginning of  period
cal_period_end = '2005-12-31 23:00:00' # end of period: one year is recommended
# Time period of the model simulation
sim_period_start = '2001-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2020-11-01 00:00:00'

# output
output_path = working_directory + "Output/" + data_csv[:15] + sim_period_start[:4] + "_" + sim_period_end[:4] + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "/"
#os.mkdir(output_path) # creates new folder for each model run with timestamp

glacier_area = 2.566
catchment_area = 46.232

# Downscaling the temperature and precipitation to glacier altitude for the DDM
lapse_rate_temperature = -0.006 # K/m
lapse_rate_precipitation = 0
height_diff_catchment = -504 # height data is 3864 m, catchment mean is 3360 glacier mean is 4042m
height_diff_glacier = 178

cal_exclude = True # Include or exclude the calibration period
plot_frequency = "D" # possible options are "D" (daily), "W" (weekly), "M" (monthly) or "Y" (yearly)
plot_frequency_long = "Daily" # Daily, Weekly, Monthly or Yearly
plot_save = False # saves plot in folder, otherwise just shows it in Python

## Data input preprocessing
print('---')
print('Starting MATILDA model run')
#print('Read input netcdf file %s' % (cosipy_nc))
print('Read input csv file %s' % (data_csv))
print('Read observation data %s' % (observation_data))
# Import necessary input: cosipy.nc, cosipy.csv and runoff observation data
#ds = xr.open_dataset(input_path_data + cosipy_nc)
df = pd.read_csv(input_path_data + data_csv)
obs = pd.read_csv(input_path_observations + observation_data)
cmip_factors = pd.read_csv(cmip_data)

print("Spin up period between " + str(cal_period_start) + " and "  + str(cal_period_end))
print("Simulation period between " + str(sim_period_start) + " and "  + str(sim_period_end))
df = dataformatting.data_preproc(df, cal_period_start, sim_period_end) # formatting the input to right format
obs = dataformatting.data_preproc(obs, cal_period_start, sim_period_end)
#obs = obs.tz_localize('Asia/Bishkek')

df = df.drop(columns="Unnamed: 0")

cmip_factors_26_temp = cmip_factors[cmip_factors["scenario"] == "temp_26"]
cmip_factors_26_prec = cmip_factors[cmip_factors["scenario"] == "prec_26"]
cmip_factors_45_temp = cmip_factors[cmip_factors["scenario"] == "temp_45"]
cmip_factors_45_prec = cmip_factors[cmip_factors["scenario"] == "prec_45"]
cmip_factors_85_temp = cmip_factors[cmip_factors["scenario"] == "temp_85"]
cmip_factors_85_prec = cmip_factors[cmip_factors["scenario"] == "prec_85"]

cmip_factors_26_temp = cmip_factors_26_temp.reset_index()
cmip_factors_26_prec = cmip_factors_26_prec.reset_index()
cmip_factors_45_temp = cmip_factors_45_temp.reset_index()
cmip_factors_45_prec = cmip_factors_45_prec.reset_index()
cmip_factors_85_temp = cmip_factors_85_temp.reset_index()
cmip_factors_85_prec = cmip_factors_85_prec.reset_index()

df["month"] = df.index.month

df_26_2040 = df.copy()
df_26_2060 = df.copy()
df_26_2080 = df.copy()
df_26_2100 = df.copy()
df_45_2040 = df.copy()
df_45_2060 = df.copy()
df_45_2080 = df.copy()
df_45_2100 = df.copy()
df_85_2040 = df.copy()
df_85_2060 = df.copy()
df_85_2080 = df.copy()
df_85_2100 = df.copy()

for i in range(1, 13):
    df_26_2040["T2"] = np.where(df_26_2040["month"] == i, df_26_2040["T2"] + cmip_factors_26_temp.loc[i-1, "diff_hist_2040"], df_26_2040["T2"])
    df_26_2040["RRR"] = np.where(df_26_2040["month"] == i, df_26_2040["RRR"] * cmip_factors_26_prec.loc[i - 1, "prec_fact_2040"], df_26_2040["RRR"])
for i in range(1, 13):
    df_26_2060["T2"] = np.where(df_26_2060["month"] == i, df_26_2060["T2"] + cmip_factors_26_temp.loc[i-1, "diff_hist_2060"], df_26_2060["T2"])
    df_26_2060["RRR"] = np.where(df_26_2060["month"] == i, df_26_2060["RRR"] * cmip_factors_26_prec.loc[i - 1, "prec_fact_2060"], df_26_2060["RRR"])
for i in range(1, 13):
    df_26_2080["T2"] = np.where(df_26_2080["month"] == i, df_26_2080["T2"] + cmip_factors_26_temp.loc[i-1, "diff_hist_2080"], df_26_2080["T2"])
    df_26_2080["RRR"] = np.where(df_26_2080["month"] == i, df_26_2080["RRR"] * cmip_factors_26_prec.loc[i - 1, "prec_fact_2080"], df_26_2080["RRR"])
for i in range(1, 13):
    df_26_2100["T2"] = np.where(df_26_2100["month"] == i, df_26_2100["T2"] + cmip_factors_26_temp.loc[i-1, "diff_hist_2100"], df_26_2100["T2"])
    df_26_2100["RRR"] = np.where(df_26_2100["month"] == i, df_26_2100["RRR"] * cmip_factors_26_prec.loc[i - 1, "prec_fact_2100"], df_26_2100["RRR"])
for i in range(1, 13):
    df_45_2040["T2"] = np.where(df_45_2040["month"] == i, df_45_2040["T2"] + cmip_factors_45_temp.loc[i-1, "diff_hist_2040"], df_45_2040["T2"])
    df_45_2040["RRR"] = np.where(df_45_2040["month"] == i, df_45_2040["RRR"] * cmip_factors_45_prec.loc[i - 1, "prec_fact_2040"], df_45_2040["RRR"])
for i in range(1, 13):
    df_45_2060["T2"] = np.where(df_45_2060["month"] == i, df_45_2060["T2"] + cmip_factors_45_temp.loc[i-1, "diff_hist_2060"], df_45_2060["T2"])
    df_45_2060["RRR"] = np.where(df_45_2060["month"] == i, df_45_2060["RRR"] * cmip_factors_45_prec.loc[i - 1, "prec_fact_2060"], df_45_2060["RRR"])
for i in range(1, 13):
    df_45_2080["T2"] = np.where(df_45_2080["month"] == i, df_45_2080["T2"] + cmip_factors_45_temp.loc[i-1, "diff_hist_2080"], df_45_2080["T2"])
    df_45_2080["RRR"] = np.where(df_45_2080["month"] == i, df_45_2080["RRR"] * cmip_factors_45_prec.loc[i - 1, "prec_fact_2080"], df_45_2080["RRR"])
for i in range(1, 13):
    df_45_2100["T2"] = np.where(df_45_2100["month"] == i, df_45_2100["T2"] + cmip_factors_45_temp.loc[i-1, "diff_hist_2100"], df_45_2100["T2"])
    df_45_2100["RRR"] = np.where(df_45_2100["month"] == i, df_45_2100["RRR"] * cmip_factors_45_prec.loc[i - 1, "prec_fact_2100"], df_45_2100["RRR"])
for i in range(1, 13):
    df_85_2040["T2"] = np.where(df_85_2040["month"] == i, df_85_2040["T2"] + cmip_factors_85_temp.loc[i-1, "diff_hist_2040"], df_85_2040["T2"])
    df_85_2040["RRR"] = np.where(df_85_2040["month"] == i, df_85_2040["RRR"] * cmip_factors_85_prec.loc[i - 1, "prec_fact_2040"], df_85_2040["RRR"])
for i in range(1, 13):
    df_85_2060["T2"] = np.where(df_85_2060["month"] == i, df_85_2060["T2"] + cmip_factors_85_temp.loc[i-1, "diff_hist_2060"], df_85_2060["T2"])
    df_85_2060["RRR"] = np.where(df_85_2060["month"] == i, df_85_2060["RRR"] * cmip_factors_85_prec.loc[i - 1, "prec_fact_2060"], df_85_2060["RRR"])
for i in range(1, 13):
    df_85_2080["T2"] = np.where(df_85_2080["month"] == i, df_85_2080["T2"] + cmip_factors_85_temp.loc[i-1, "diff_hist_2080"], df_85_2080["T2"])
    df_85_2080["RRR"] = np.where(df_85_2080["month"] == i, df_85_2080["RRR"] * cmip_factors_85_prec.loc[i - 1, "prec_fact_2080"], df_85_2080["RRR"])
for i in range(1, 13):
    df_85_2100["T2"] = np.where(df_85_2100["month"] == i, df_85_2100["T2"] + cmip_factors_85_temp.loc[i-1, "diff_hist_2100"], df_85_2100["T2"])
    df_85_2100["RRR"] = np.where(df_85_2100["month"] == i, df_85_2100["RRR"] * cmip_factors_85_prec.loc[i - 1, "prec_fact_2100"], df_85_2100["RRR"])


# Downscaling the dataframe to the glacier height
df_DDM = dataformatting.glacier_downscaling(df, height_diff=height_diff_glacier, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)
df = dataformatting.glacier_downscaling(df, height_diff=height_diff_catchment, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)

