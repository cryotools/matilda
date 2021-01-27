## Running all the required functions
from datetime import datetime
import os
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA import dataformatting # prepare and format the input and output data
from MATILDA import DDM # importing the DDM model functions
from MATILDA import HBV # importing the HBV model function
from MATILDA import stats, plots # importing functions for statistical analysis and plotting

## Model configuration
# Directories
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package/"
input_path_data = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/"
input_path_observations = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"

data_csv = "no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
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
plot_frequency = "M" # possible options are "D" (daily), "W" (weekly), "M" (monthly) or "Y" (yearly)
plot_frequency_long = "Monthly" # Daily, Weekly, Monthly or Yearly
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
df = df.resample("D").agg({"T2":"mean", "RRR":"sum"})

cmip_output = pd.DataFrame(index=df.index)

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

df_hist = df.copy()
df_hist.name = "df_hist"

df_hist["T2"] = df_hist["T2"] +1

df["month"] = df.index.month

df_26_2040 = df.copy()
df_26_2040.name = "df_26_2040"
df_26_2060 = df.copy()
df_26_2060.name = "df_26_2060"
df_26_2080 = df.copy()
df_26_2080.name = "df_26_2080"
df_26_2100 = df.copy()
df_26_2100.name = "df_26_2100"
df_45_2040 = df.copy()
df_45_2040.name = "df_45_2040"
df_45_2060 = df.copy()
df_45_2060.name = "df_45_2060"
df_45_2080 = df.copy()
df_45_2080.name = "df_45_2080"
df_45_2100 = df.copy()
df_45_2100.name = "df_45_2100"
df_85_2040 = df.copy()
df_85_2040.name = "df_85_2040"
df_85_2060 = df.copy()
df_85_2060.name = "df_85_2060"
df_85_2080 = df.copy()
df_85_2080.name = "df_85_2080"
df_85_2100 = df.copy()
df_85_2100.name = "df_85_2100"

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

##
list_cmip = [df_hist, df_26_2040, df_26_2060, df_26_2080, df_26_2100, df_45_2040, df_45_2060, df_45_2080, df_45_2100, df_85_2040, df_85_2060, df_85_2080, df_85_2100]

for i in list_cmip:
# Downscaling the dataframe to the glacier height
    df_DDM = dataformatting.glacier_downscaling(i, height_diff=height_diff_glacier, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)
    df = dataformatting.glacier_downscaling(i, height_diff=height_diff_catchment, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)

    degreedays_ds = DDM.calculate_PDD(df_DDM)
    output_DDM, parameter_DDM = DDM.calculate_glaciermelt(degreedays_ds, pdd_factor_snow=5.5, pdd_factor_ice=8.5, temp_snow=-0.5) # output in mm, parameter adjustment possible
    output_DDM["Q_DDM"] = output_DDM["Q_DDM"]*(glacier_area/catchment_area) # scaling glacier melt to glacier area
    output_hbv, parameter_HBV = HBV.hbv_simulation(df, cal_period_start, cal_period_end, parTT=-0.5, parPERC=2.5, parFC=150, parUZL=60) # output in mm, individual parameters can be set here

    output = pd.concat([output_hbv, output_DDM], axis=1)
    output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]
    cmip_output[i.name]= output["Q_Total"]

cmip_output.to_csv("/home/ana/Desktop/cmip_output.csv")

## Glacier
list_cmip_60 = [df_26_2060, df_45_2060, df_85_2060]
cmip_output_glacier = pd.DataFrame(index=df.index)

for i in list_cmip_60:
# Downscaling the dataframe to the glacier height
    df_DDM = dataformatting.glacier_downscaling(i, height_diff=236, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)
    df = dataformatting.glacier_downscaling(i, height_diff=height_diff_catchment, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)

    degreedays_ds = DDM.calculate_PDD(df_DDM)
    output_DDM, parameter_DDM = DDM.calculate_glaciermelt(degreedays_ds, pdd_factor_snow=5.5, pdd_factor_ice=8.5, temp_snow=-0.5) # output in mm, parameter adjustment possible
    output_DDM["Q_DDM"] = output_DDM["Q_DDM"]*((glacier_area*0.5)/catchment_area) # scaling glacier melt to glacier area
    output_hbv, parameter_HBV = HBV.hbv_simulation(df, cal_period_start, cal_period_end, parTT=-0.5, parPERC=2.5, parFC=150, parUZL=60) # output in mm, individual parameters can be set here

    output = pd.concat([output_hbv, output_DDM], axis=1)
    output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]
    cmip_output_glacier[i.name]= output["Q_Total"]

# CMIP 8.5 2060-2080
df_DDM = dataformatting.glacier_downscaling(df_85_2080, height_diff=286, lapse_rate_temperature=lapse_rate_temperature,
                                            lapse_rate_precipitation=lapse_rate_precipitation)
df = dataformatting.glacier_downscaling(df_85_2080, height_diff=height_diff_catchment, lapse_rate_temperature=lapse_rate_temperature,
                                        lapse_rate_precipitation=lapse_rate_precipitation)

degreedays_ds = DDM.calculate_PDD(df_DDM)
output_DDM, parameter_DDM = DDM.calculate_glaciermelt(degreedays_ds, pdd_factor_snow=5.5, pdd_factor_ice=8.5, temp_snow=-0.5)  # output in mm, parameter adjustment possible
output_DDM["Q_DDM"] = output_DDM["Q_DDM"] * ((glacier_area * 0.4) / catchment_area)  # scaling glacier melt to glacier area
output_hbv, parameter_HBV = HBV.hbv_simulation(df, cal_period_start, cal_period_end, parTT=-0.5, parPERC=2.5, parFC=150,parUZL=60)  # output in mm, individual parameters can be set here

output = pd.concat([output_hbv, output_DDM], axis=1)
output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]
cmip_output_glacier["df_85_2080"] = output["Q_Total"]

# CMIP 8.5 2080-2100
df_DDM = dataformatting.glacier_downscaling(df_85_2100, height_diff=336, lapse_rate_temperature=lapse_rate_temperature,
                                            lapse_rate_precipitation=lapse_rate_precipitation)
df = dataformatting.glacier_downscaling(df_85_2100, height_diff=height_diff_catchment, lapse_rate_temperature=lapse_rate_temperature,
                                        lapse_rate_precipitation=lapse_rate_precipitation)

degreedays_ds = DDM.calculate_PDD(df_DDM)
output_DDM, parameter_DDM = DDM.calculate_glaciermelt(degreedays_ds, pdd_factor_snow=5.5, pdd_factor_ice=8.5, temp_snow=-0.5)  # output in mm, parameter adjustment possible
output_DDM["Q_DDM"] = output_DDM["Q_DDM"] * ((glacier_area * 0.3) / catchment_area)  # scaling glacier melt to glacier area
output_hbv, parameter_HBV = HBV.hbv_simulation(df, cal_period_start, cal_period_end, parTT=-0.5, parPERC=2.5, parFC=150,parUZL=60)  # output in mm, individual parameters can be set here

output = pd.concat([output_hbv, output_DDM], axis=1)
output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]
cmip_output_glacier["df_85_2100"] = output["Q_Total"]

cmip_output_glacier.to_csv("/home/ana/Desktop/cmip_output_glacier.csv")

## Plots
cmip_output_monthly = cmip_output.resample("M").agg("sum")
cmip_output_monthly["month"] = cmip_output_monthly.index.month
cmip_output_monthly_mean = cmip_output_monthly.groupby(["month"]).mean()
cmip_output_monthly_mean["month"] = cmip_output_monthly_mean.index



fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharey=True, figsize=(10, 6))
gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0, 0])
ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2040"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2040"], c="#009E73", linewidth=1.2, label="RCP 4.5")
ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2040"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
ax2 = plt.subplot(gs[0,1], sharey=ax1)
ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2060"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2060"], c="#009E73", linewidth=1.2, label="RCP 4.5")
ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2060"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
ax3 = plt.subplot(gs[1,0], sharey=ax1)
ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2080"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2080"], c="#009E73", linewidth=1.2, label="RCP 4.5")
ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2080"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
ax4 = plt.subplot(gs[1,1], sharey=ax1)
ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2100"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2100"], c="#009E73", linewidth=1.2, label="RCP 4.5")
ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2100"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
#ax1.legend(), ax2.legend(), ax3.legend(), ax4.legend()
ax1.xaxis.set_ticks(np.arange(2, 12, 2)), ax2.xaxis.set_ticks(np.arange(2, 12, 2)), ax3.xaxis.set_ticks(np.arange(2, 12, 2)); ax4.xaxis.set_ticks(np.arange(2, 12, 2))
fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9), ax4.set_ylabel("[mm]", fontsize=9)
ax1.set_title("Period of 2020 - 2040", fontsize=9)
ax2.set_title("Period of 2040 - 2060", fontsize=9)
ax3.set_title("Period of 2060 - 2080", fontsize=9)
ax4.set_title("Period of 2080 - 2100", fontsize=9)
#plt.show()
plt.savefig("/home/ana/Desktop/CMIP_mean_monthly_runoff.png")


output_19_20 = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package/Output/no182_ERA5_Land2019_2020_2021-01-27_16:14:34/model_output_2018-2020.csv")
output_19_20 = output_19_20.set_index("TIMESTAMP")
output_19_20.index = pd.to_datetime(output_19_20.index)
output_19_20 = output_19_20["2019-01-01":"2020-12-31"]
output_19_20["plot"] = 0


plt.figure(figsize=(10,6))
plt.plot(output_19_20.index.to_pydatetime(), output_19_20["Qobs"], c="#D55E00", label="Observations", linewidth=1)
plt.plot(output_19_20.index.to_pydatetime(), output_19_20["Q_Total"], c="k", label="MATILDA", linewidth=0.5)
plt.fill_between(output_19_20.index.to_pydatetime(), output_19_20["plot"], output_19_20["Q_HBV"],color='#009E73',alpha=.5, label=" MATILDA HBV")
plt.fill_between(output_19_20.index.to_pydatetime(), output_19_20["Q_HBV"], output_19_20["Q_Total"],color='#CC79A7',alpha=.5, label="MATILDA DDM")
plt.legend()
plt.ylabel("[mm]", fontsize=9)
#plt.show()
plt.savefig("/home/ana/Desktop/MATILDA_output_2019-20.png")


cmip_output_glacier_monthly = cmip_output_glacier.resample("M").agg("sum")
cmip_output_glacier_monthly["month"] = cmip_output_glacier_monthly.index.month
cmip_output_glacier_monthly_mean = cmip_output_glacier_monthly.groupby(["month"]).mean()
cmip_output_glacier_monthly_mean["month"] = cmip_output_glacier_monthly_mean.index

plt.figure(figsize=(10,6))
plt.plot(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_26_2060"], c="#0072B2", label="RCP 2.6 - 2040-2060")
plt.plot(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_45_2060"], c="k", label="RCP 4.5 - 2040-2060")
plt.plot(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_85_2060"], c="#D55E00",label="RCP 8.5 - 2040-2060")
plt.plot(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_85_2080"], c="#009E73", label="RCP 8.5 - 2060-2080")
plt.plot(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_85_2100"], c="#CC79A7", label="RCP 8.5 - 2080-2100")
plt.legend()
plt.ylabel("[mm]", fontsize=9)
#plt.show()
plt.savefig("/home/ana/Desktop/glacier_loss_runs.png")

fig, ax = plt.subplots()
ax.bar(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_26_2060"], label="RCP 2.6 - 2040-2060")
ax.bar(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_45_2060"], label="RCP 4.5 - 2040-2060")
ax.bar(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_85_2060"], label="RCP 8.5 - 2040-2060")
ax.bar(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_85_2080"],  label="RCP 8.5 - 2060-2080")
ax.bar(cmip_output_glacier_monthly_mean["month"], cmip_output_glacier_monthly_mean["df_85_2100"], label="RCP 8.5 - 2080-2100")
plt.legend()
plt.ylabel("[mm]", fontsize=9)
plt.show()