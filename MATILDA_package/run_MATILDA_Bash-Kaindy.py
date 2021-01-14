# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## Running all the required functions
from datetime import datetime
import os
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
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

cosipy_nc = ""
data_csv = "no182ERA5_Land_2018_2019_down.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
observation_data = "runoff_bashkaindy_2019.csv" # Daily Runoff Observations in mm

# output
output_path = working_directory + "Output/" + data_csv[:-9] + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "/"
os.mkdir(output_path) # creates new folder for each model run with timestamp

# Additional information
# Time period for the spin up
cal_period_start = '2019-01-01 00:00:00' # beginning of  period
cal_period_end = '2019-12-31 23:00:00' # end of period: one year is recommended
# Time period of the model simulation
sim_period_start = '2019-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2019-12-31 23:00:00'

glacier_area = 2.566
catchment_area = 46.232

# Downscaling the temperature and precipitation to glacier altitude for the DDM
lapse_rate_temperature = -0.006 # K/m
lapse_rate_precipitation = 0
height_diff = 682 # height difference between AWS (4025) and glacier (4036) in m

cal_exclude = False # Include or exclude the calibration period
plot_frequency = "D" # possible options are "D" (daily), "W" (weekly), "M" (monthly) or "Y" (yearly)
plot_frequency_long = "Daily" # Daily, Weekly, Monthly or Yearly
plot_save = True # saves plot in folder, otherwise just shows it in Python
cosipy = False  # usage of COSIPY input

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

print("Spin up period between " + str(cal_period_start) + " and "  + str(cal_period_end))
print("Simulation period between " + str(sim_period_start) + " and "  + str(sim_period_end))
df = dataformatting.data_preproc(df, cal_period_start, sim_period_end) # formatting the input to right format
#ds = dataformatting.data_preproc(ds, cal_period_start, sim_period_end)
obs = dataformatting.data_preproc(obs, cal_period_start, sim_period_end)
obs = obs.tz_localize('Asia/Bishkek')

# Downscaling the dataframe to the glacier height
df_DDM = dataformatting.glacier_downscaling(df, height_diff=height_diff, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)

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
print("Calculating the positive degree days")
degreedays_ds = DDM.calculate_PDD(df_DDM)
print("Calculating melt with the DDM")
# include either downscaled glacier dataframe or dataset with mask
# Calculating runoff and melt
output_DDM, parameter_DDM = DDM.calculate_glaciermelt(degreedays_ds, temp_snow=-1) # output in mm, parameter adjustment possible
output_DDM["Q_DDM"] = output_DDM["Q_DDM"]*(glacier_area/catchment_area) # scaling glacier melt to glacier area
print("Finished running the DDM")

## HBV model
print("Running the HBV model")
# Runoff calculations for the catchment with the HBV model
output_hbv, parameter_HBV = HBV.hbv_simulation(df, cal_period_start, cal_period_end, parTT=-1) # output in mm, individual parameters can be set here
print("Finished running the HBV")
## Output postprocessing
output_hbv["Q_HBV"] = output_hbv["Q_HBV"] - (output_DDM["DDM_snow_melt_rate"]*(glacier_area/catchment_area))
output = dataformatting.output_postproc(output_hbv, output_DDM, obs)

nash_sut = stats.NS(output["Qobs"], output["Q_Total"]) # Nash–Sutcliffe model efficiency coefficient
if nash_sut == "error":
    print("ERROR. The Nash–Sutcliffe model efficiency coefficient is outside the range of -1 to 1")
else:
    print("The Nash–Sutcliffe model efficiency coefficient of the MATILDA run is " + str(round(nash_sut, 2)))

print("Writing the output csv to disc")
output = output.fillna(0)
output.to_csv(output_path + "model_output_" +str(cal_period_start[:4])+"-"+str(sim_period_end[:4]+".csv"))

parameter = dataformatting.output_parameter(parameter_HBV, parameter_DDM)
parameter.to_csv(output_path + "model_parameter.csv")
## Statistical analysis
# Calibration period included in the statistical analysis
if cal_exclude == True:
    output_calibration = output[~(output.index < cal_period_end)]
else:
    output_calibration = output.copy()

# Daily, weekly, monthly or yearly output
plot_data = dataformatting.plot_data(output, plot_frequency, cal_period_start, sim_period_end)

stats_output = stats.create_statistics(output_calibration)
stats_output.to_csv(output_path + "model_stats_" +str(output_calibration.index.values[1])[:4]+"-"+str(output_calibration.index.values[-1])[:4]+".csv")
print("Output overview")
print(stats_output[["T2", "RRR", "PE", "Q_DDM", "Qobs", "Q_Total"]])
## Cosipy comparison
if cosipy == True:
    output_cosipy = dataformatting.output_cosipy(output, ds)
    output_cosipy.to_csv(output_path + "cosipy_comparison_output_" + str(cal_period_start[:4]) + "-" + str(sim_period_end[:4] + ".csv"))

    nash_sut_cosipy = stats.NS(output_cosipy["Qobs"], output_cosipy["Q_COSIPY"])

    stats_cosipy = stats.create_statistics(output_cosipy)
    stats_cosipy.to_csv(output_path + "cosipy_comparison_stats_" + str(output_calibration.index.values[1])[:4] + "-" + str(
        output_calibration.index.values[-1])[:4] + ".csv")
    plot_data_cosipy = dataformatting.plot_data_cosipy(output_cosipy, plot_frequency, cal_period_start, sim_period_end)

    fig3 = plots.plot_cosipy(plot_data_cosipy, plot_frequency_long, nash_sut, nash_sut_cosipy)
    if plot_save == False:
        plt.show()
    else:
        if str(plot_data_cosipy.index.values[1])[:4] == str(plot_data_cosipy.index.values[-1])[:4]:
            plt.savefig(output_path + "COSIPY_output_" + + str(plot_data_cosipy.index.values[-1])[:4] + ".png")
        else:
            plt.savefig(output_path + "COSIPY_output_" + str(plot_data_cosipy.index.values[1])[:4] + "-" + str(
            plot_data_cosipy.index.values[-1])[:4] + ".png")

## Plotting the output data
# Plot the meteorological data
fig = plots.plot_meteo(plot_data, plot_frequency_long)
if plot_save == False:
	plt.show()
else:
    if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
        plt.savefig(output_path + "meteorological_data_" + str(plot_data.index.values[-1])[:4]+".png")
    else:
	    plt.savefig(output_path + "meteorological_data_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot the runoff data
fig1 = plots.plot_runoff(plot_data, plot_frequency_long, nash_sut)
if plot_save == False:
	plt.show()
else:
    if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
        plt.savefig(output_path + "model_runoff_" + str(plot_data.index.values[-1])[:4]+".png")
    else:
	    plt.savefig(output_path + "model_runoff_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot the HBV paramters
fig2 = plots.plot_hbv(plot_data, plot_frequency_long)
if plot_save == False:
	plt.show()
else:
    if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
        plt.savefig(output_path + "HBV_output_" + str(plot_data.index.values[-1])[:4]+".png")
    else:
	    plt.savefig(output_path + "HBV_output_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

print('Saved plots of meteorological and runoff data to disc')
print("End of model run")
print('---')


## Tests
import numpy as np
obs.loc[obs['temp'] > 50, 'temp'] = np.nan


plt.plot(plot_data.index.to_pydatetime(), (plot_data["T2"]), c="#d7191c")
plt.plot(obs.index.to_pydatetime(), obs["temp"], c="#008837")
plt.xlabel("Date", fontsize=9)
plt.show()

minikin = pd.read_csv("/home/ana/Downloads/cognac_glacier_minikin_18_19.csv")
minikin.columns = ['datetime', 'G', 'temp', 'hum']
minikin.set_index(pd.to_datetime(minikin.datetime), inplace=True)
minikin = minikin.drop(['datetime'], axis=1)
minikin = minikin.shift(-2, axis=0)
minikin = minikin.resample(plot_frequency).agg({"temp": "mean"})

df_DDM = df_DDM.resample(plot_frequency).agg({"T2":"mean"})
plt.plot(df_DDM.index.to_pydatetime(), (df_DDM["T2"]), c="#d7191c")
plt.plot(minikin.index.to_pydatetime(), minikin["temp"], c="#008837")
plt.xlabel("Date", fontsize=9)
plt.show()