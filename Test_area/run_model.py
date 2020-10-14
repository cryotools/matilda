# -*- coding: UTF-8 -*-
"""
The model is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of the
glacierized catchments.
This file uses the input files created by the COSIPY-utility "aws2cosipy" as forcing data and additional
observation runoff data to validate it.
"""
## Running all the model functions
from datetime import datetime
import os
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from finalmodel import DDM # importing the DDM model functions
from finalmodel import HBV # importing the HBV model function
from finalmodel import stats, plots

## Model configuration
# Directories
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/Final_Model/"
input_path_cosipy = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/best_cosipyrun_no1/best_cosipyrun_no1_2011-18/"
input_path_observations = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/glacierno1/hydro/"

cosipy_nc = "best_cosipy_output_no1_2011-18.nc"
data_csv = "best_cosipy_input_no1_2011-18.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
observation_data = "daily_observations_2011-18.csv" # Daily Runoff Observations in mm

# output
output_path = working_directory + "Output_package/" + cosipy_nc[:-3] + ">>" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "/"
os.mkdir(output_path) # creates new folder for each model run with timestamp

# Additional information
# Time period for the spin up
cal_period_start = '2011-01-01 00:00:00' # beginning of  period
cal_period_end = '2011-12-31 23:00:00' # end of period: one year is recommended
# Time period of the model simulation
sim_period_start = '2012-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2018-12-31 23:00:00'

# Downscaling the temperature and precipitation to glacier altitude for the DDM
lapse_rate_temperature = -0.006 # K/m
lapse_rate_precipitation = 0
height_diff = 21 # height difference between AWS (4025) and glacier (4036) in m

cal_exclude = False # Include or exclude the calibration period
plot_frequency = "weekly" # possible options are daily, weekly, monthly or yearly
plot_save = True # saves plot in folder, otherwise just shows it in Python
cosipy = True # usage of COSIPY input

## Data input preprocessing
print('---')
print('Read input netcdf file %s' % (cosipy_nc))
print('Read input csv file %s' % (data_csv))
print('Read observation data %s' % (observation_data))
# Import necessary input: cosipy.nc, cosipy.csv and runoff observation data
ds = xr.open_dataset(input_path_cosipy + cosipy_nc)
df = pd.read_csv(input_path_cosipy + data_csv)
obs = pd.read_csv(input_path_observations + observation_data)

print("Spin up period between " + str(cal_period_start) + " and "  + str(cal_period_end))
print("Simulation period between " + str(sim_period_start) + " and "  + str(sim_period_end))
# adjust time
ds = ds.sel(time=slice(cal_period_start, sim_period_end))
df.set_index('TIMESTAMP', inplace=True) # set date column as index
df.index = pd.to_datetime(df.index)
df = df[cal_period_start: sim_period_end]
obs.set_index('Date', inplace=True)
obs.index = pd.to_datetime(obs.index) # set date column as index
obs = obs[cal_period_start: sim_period_end]

# Downscaling the dataframe to the glacier height
df_DDM = df.copy()
df_DDM["T2"] = df_DDM["T2"] + height_diff * float(lapse_rate_temperature)
df_DDM["RRR"] = df_DDM["RRR"] + height_diff * float(lapse_rate_precipitation)

# adjusting the variable units: T2: K to °C
df["T2"] = df["T2"] - 273.15
df_DDM["T2"] = df_DDM["T2"] - 273.15
ds["T2"] = ds["T2"] - 273.15

## DDM model
print("Running the degree day model")
"""Parameter DDM
    'pdd_factor_snow': 2.8, # according to Huintjes et al. 2010  [5.7 mm per day per Celsius according to Hock 2003]
    'pdd_factor_ice': 5.6,  # according to Huintjes et al. 2010 [7.4 mm per day per Celsius according to Hock 2003]
    'temp_snow': 0.0,
    'temp_rain': 2.0,
    'refreeze_snow': 0.0,
    'refreeze_ice': 0.0}"""

# Calculating the positive degree days
degreedays_ds = DDM.calculate_PDD(ds) # include either downscaled glacier dataframe or dataset with mask
# Calculating runoff and melt
output_DDM = DDM.calculate_glaciermelt(degreedays_ds) # output in mm, parameter adjustment possible
print(output_DDM.head(5))
## HBV model
print("Running the HBV model")
# Runoff calculations for the catchment with the HBV model
output_hbv = HBV.hbv_simulation(df, cal_period_start, cal_period_end) # output in mm, individual parameters can be set here
print(output_hbv.head(5))

## Output postprocessing
output = pd.concat([output_hbv, output_DDM], axis=1)
output = pd.concat([output, obs], axis=1)
output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]

nash_sut = stats.NS(output["Qobs"], output["Q_Total"]) # Nash–Sutcliffe model efficiency coefficient
print("The Nash–Sutcliffe model efficiency coefficient of the total model is " + str(round(nash_sut, 2)))

print("Writing the output csv to disc")
output = output.fillna(0)
output.to_csv(output_path + "model_output_" +str(cal_period_start[:4])+"-"+str(sim_period_end[:4]+".csv"))

## Statistical analysis
# Calibration period included in the statistical analysis
if cal_exclude == True:
    output_calibration = output[~(output.index < cal_period_end)]
else:
    output_calibration = output.copy()

# Daily, monthly or yearly output
if plot_frequency == "daily":
    plot_data = output_calibration.copy()
elif plot_frequency == "weekly":
    plot_data = output_calibration.resample("W").agg(
        {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", "Qobs": "sum", \
         "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
         "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})
elif plot_frequency == "monthly":
    plot_data = output_calibration.resample("M").agg(
        {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", "Qobs": "sum", \
         "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
         "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})
elif plot_frequency == "yearly":
    plot_data = output_calibration.resample("Y").agg(
        {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", "Qobs": "sum", \
         "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
         "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})

stats_output = stats.create_statistics(output_calibration)
stats_output.to_csv(output_path + "model_stats_" +str(output_calibration.index.values[1])[:4]+"-"+str(output_calibration.index.values[-1])[:4]+".csv")

## Cosipy comparison
if cosipy == True:
    output_cosipy = output[{"Qobs", "Q_Total", "DDM_smb", "DDM_total_melt"}]
    cosipy_runoff = ds.Q.mean(dim=["lat", "lon"])
    cosipy_smb = ds.surfMB.mean(dim=["lat", "lon"])
    #cosipy_smb = cosipy_smb.resample(time="D").sum(dim="time")
    cosipy_melt = ds.surfM.mean(dim=["lat", "lon"])
    #cosipy_melt = cosipy_melt.resample(time="D").sum(dim="time")
    output_cosipy["Q_COSIPY"] = cosipy_runoff.to_dataframe().Q.resample('D').sum()*1000
    output_cosipy["COSIPY_smb"] = cosipy_smb.to_dataframe().surfMB.resample('D').sum()*1000
    output_cosipy["COSIPY_melt"] = cosipy_melt.to_dataframe().surfM.resample('D').sum()*1000
    output_cosipy = output_cosipy.round(3)
    output_cosipy.to_csv(output_path + "cosipy_comparison_output_" + str(cal_period_start[:4]) + "-" + str(sim_period_end[:4] + ".csv"))

    nash_sut_cosipy = stats.NS(output_cosipy["Qobs"], output_cosipy["Q_COSIPY"])

    stats_cosipy = stats.create_statistics(output_cosipy)
    stats_cosipy.to_csv(output_path + "cosipy_comparison_stats_" + str(output_calibration.index.values[1])[:4] + "-" + str(
        output_calibration.index.values[-1])[:4] + ".csv")
    if plot_frequency == "daily":
        plot_data_cosipy = output_cosipy.copy()
    elif plot_frequency == "weekly":
        plot_data_cosipy = output_cosipy.resample("W").agg(
            {"Qobs": "sum", "Q_Total": "sum", "Q_COSIPY": "sum", "DDM_smb":"sum", "DDM_total_melt":"sum", \
             "COSIPY_smb":"sum", "COSIPY_melt":"sum"})
    elif plot_frequency == "monthly":
        plot_data_cosipy = output_cosipy.resample("M").agg(
            {"Qobs": "sum", "Q_Total": "sum", "Q_COSIPY": "sum", "DDM_smb":"sum", "DDM_total_melt":"sum", \
             "COSIPY_smb":"sum", "COSIPY_melt":"sum"})
    elif plot_frequency == "yearly":
        plot_data_cosipy = output_cosipy.resample("Y").agg(
            {"Qobs": "sum", "Q_Total": "sum", "Q_COSIPY": "sum", "DDM_smb":"sum", "DDM_total_melt":"sum", \
             "COSIPY_smb":"sum", "COSIPY_melt":"sum" })

    fig3 = plots.plot_cosipy(plot_data_cosipy, nash_sut, nash_sut_cosipy)
    if plot_save == False:
        plt.show()
    else:
        plt.savefig(output_path + "COSIPY_output_" + str(plot_data_cosipy.index.values[1])[:4] + "-" + str(
            plot_data_cosipy.index.values[-1])[:4] + ".png")

## Plotting the output data
# Plot the meteorological data
fig = plots.plot_meteo(plot_data)
if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "meteorological_data_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot the runoff data
fig1 = plots.plot_runoff(plot_data, nash_sut)
if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "model_runoff_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot the HBV paramters
fig2 = plots.plot_hbv(plot_data)
if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "HBV_output_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

print('Saved plots of meteorological and runoff data to disc')
print("End of model run")
print('---')
