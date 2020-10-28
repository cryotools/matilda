# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## Running all the required functions
from datetime import datetime
import pandas as pd
import numpy as np
import copy
import matplotlib.pyplot as plt
from MATILDA import DDM # importing the DDM model functions
from MATILDA import HBV # importing the HBV model function

## Model configuration
# Directories
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package/"
input_path_data = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/best_cosipyrun_no1/"

data_csv = "best_cosipy_input_no1_2000-20.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)

# Additional information
# Time period for the spin up
cal_period_start = '2000-01-01 00:00:00' # beginning of  period
cal_period_end = '2001-12-31 23:00:00' # end of period: one year is recommended
# Time period of the model simulation
sim_period_start = '2002-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2020-07-10 03:00:00'

# Downscaling the temperature and precipitation to glacier altitude for the DDM
lapse_rate_temperature = -0.006 # K/m
lapse_rate_precipitation = 0
height_diff = 21 # height difference between AWS (4025) and glacier (4036) in m

## Data input preprocessing
df = pd.read_csv(input_path_data + data_csv)
# adjust time
df.set_index('TIMESTAMP', inplace=True) # set date column as index
df.index = pd.to_datetime(df.index)
df = df[cal_period_start: sim_period_end]

## Climate change scenarios
temp_increase = np.arange(0, 5.5, 0.5)

df_cc = df.copy()
df_cc["season"] = np.where((df_cc.index.month >= 4) & (df_cc.index.month <= 9), "summer", "winter")

dataframes = {}
for i in temp_increase:
	df = copy.deepcopy(df_cc)
	df["T2"] = np.where(df["season"] == "winter", df["T2"] + (i*1.5),  df["T2"] + (i*0.5))
	dataframes[i] = df

dataframes_glacier = copy.deepcopy(dataframes)
for i in dataframes_glacier.keys():
	dataframes_glacier[i]["T2"] = dataframes_glacier[i]["T2"] + height_diff * float(lapse_rate_temperature)

degreedays_ds_cc = {}
for i in dataframes_glacier.keys():
	degreedays_ds_cc[i] = DDM.calculate_PDD(dataframes_glacier[i])

output_DDM_cc = {}
for i in degreedays_ds_cc.keys():
	output_DDM_cc[i] = DDM.calculate_glaciermelt(degreedays_ds_cc[i])

output_HBV_cc = {}
for i in dataframes.keys():
	output_HBV_cc[i] = HBV.hbv_simulation(dataframes[i], cal_period_start, cal_period_end)

output_DDM_cc_yearly = {}
for i in output_DDM_cc.keys():
	output_DDM_cc_yearly[i] = output_DDM_cc[i].resample("Y").agg({"DDM_total_melt":"sum", "Q_DDM":"sum"})

output_HBV_cc_yearly = {}
for i in output_HBV_cc.keys():
	output_HBV_cc_yearly[i] = output_HBV_cc[i].resample("Y").agg({"T2":"mean", "Q_HBV":"sum"})

runoff_cc_yearly = {}
for i, j in zip(output_DDM_cc_yearly.keys(), output_HBV_cc_yearly.keys()):
	runoff_cc_yearly[i] = output_DDM_cc_yearly[i]["Q_DDM"] + output_HBV_cc_yearly[j]["Q_HBV"]
for i in runoff_cc_yearly.keys():
	runoff_cc_yearly[i] = np.mean(runoff_cc_yearly[i].values)

temp_cc_yearly = {}
for i in output_HBV_cc_yearly.keys():
	temp_cc_yearly[i] = np.mean(output_HBV_cc_yearly[i]["T2"].values)

melt_cc_yearly = {}
for i in output_DDM_cc_yearly.keys():
	melt_cc_yearly[i] = np.mean(output_DDM_cc_yearly[i]["DDM_total_melt"].values)

temp_cc_yearly_df = pd.DataFrame(temp_cc_yearly.items(), columns=['Increase', 'Temp'])
runoff_cc_yearly_df = pd.DataFrame(runoff_cc_yearly.items(), columns=['Increase', 'Runoff'])
melt_cc_yearly_df = pd.DataFrame(melt_cc_yearly.items(), columns=['Increase', 'Melt'])
cc_yearly_df = pd.merge(temp_cc_yearly_df, melt_cc_yearly_df)
cc_yearly_df = pd.merge(cc_yearly_df, runoff_cc_yearly_df)
cc_yearly_df["Temp"] = cc_yearly_df["Temp"] + 273.15

plt.plot(cc_yearly_df["Temp"], cc_yearly_df["Runoff"], "-ok", label="Mean annual runoff")
plt.plot(cc_yearly_df["Temp"], cc_yearly_df["Melt"], "-ob", label="Mean annual glacier melt")
plt.xlabel("Temperature [Kelvin]")
plt.ylabel("[mm]")
plt.legend()
plt.title("Model response to seasonally weighted incremental temperature increase")
#plt.show()
#plt.savefig("/home/ana/Desktop/future2.png")