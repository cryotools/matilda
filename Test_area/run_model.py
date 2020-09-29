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
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from finalmodel import DDM # importing the DDM model functions
from finalmodel import HBV # importing the HBV model function
from finalmodel import stats, plots

## Model configuration
# Directories
working_directory = "/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Final_Model/"
input_path_cosipy = "/home/phillip/Seafile/Ana-Lena_Phillip/data/input_output/input/best_cosipyrun_no1/best_cosipyrun_no1_2011-18/"
input_path_observations = "/home/phillip/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/glacierno1/hydro/"

cosipy_nc = "best_cosipy_output_no1_2011-18.nc"
data_csv = "best_cosipy_input_no1_2011-18.csv" # dataframe with columns T2 (Celsius), RRR (mm) and if possible PE (mm)
observation_data = "daily_observations_2011-18.csv" # Daily Observations in mm

# output
output_path = working_directory + "Output_package/" + cosipy_nc[:-3] + ">>" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "/"
os.mkdir(output_path)

# Additional information
area_name = "Urumqi" # specify the name of your catchment here
# Time period to calibrate initial parameters
cal_period_start = '2011-01-01 00:00:00' # beginning of the calibration period
cal_period_end = '2013-12-31 23:00:00' # end of calibration: one year is recommended
# Time period of the model simulation
sim_period_start = '2014-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2018-12-31 23:00:00'
cal_exclude = False # Include or exclude the calibration period
plot_save = True

## Data input preprocessing
print('---')
print('Read input netcdf file %s' % (cosipy_nc))
print('Read input csv file %s' % (data_csv))
print('Read observation data %s' % (observation_data))
# Import necessary input: cosipy.nc, cosipy.csv and runoff observation data
ds = xr.open_dataset(input_path_cosipy + cosipy_nc)
df = pd.read_csv(input_path_cosipy + data_csv)
obs = pd.read_csv(input_path_observations + observation_data)

print("Calibration period between " + str(cal_period_start) + " and "  + str(cal_period_end))
print("Simulation period between " + str(sim_period_start) + " and "  + str(sim_period_end))
# adjust time
ds = ds.sel(time=slice(cal_period_start, sim_period_end))
df.set_index('TIMESTAMP', inplace=True)
df.index = pd.to_datetime(df.index)
df = df[cal_period_start: sim_period_end]
obs.set_index('Date', inplace=True)
obs.index = pd.to_datetime(obs.index)
# obs = obs.sort_index()
obs = obs[cal_period_start: sim_period_end]

## DDM model
print("Running the degree day model")
# Calculating the positive degree days
degreedays_ds = DDM.calculate_PDD(ds)
# Calculating runoff and melt
output_DDM = DDM.calculate_glaciermelt(degreedays_ds) # output in mm
print(output_DDM.head(5))
## HBV model
print("Running the HBV model")
parameters_HBV = [1.0, 0.15, 250, 0.055, 0.055, 0.04, 0.7, 3.0, \
        1.5, 120, 1.0, 0.0, 5.0, 0.7, 0.05, 0.1] # Initial parameters
# Runoff calculations for the catchment with the HBV model
output_hbv = HBV.hbv_simulation(df, cal_period_start, cal_period_end, parameters_HBV) # output in mm
print(output_hbv.head(5))

## Output postprocessing
output = pd.concat([output_hbv, output_DDM], axis=1)
output = pd.concat([output, obs], axis=1)
output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]
print("Writing the output csv to disc")
output_csv = output.copy()
output_csv = output_csv.fillna(0)
output_csv.to_csv(output_path + "model_output_" +str(cal_period_start[:4])+"-"+str(sim_period_end[:4]+".csv"))

## Statistical analysis
# Calibration period included in the statistical analysis
if cal_exclude == True:
    output_calibration = output[~(output.index < cal_period_end)]
else:
    output_calibration = output.copy()

# Daily, monthly or yearly output
if plot_frequency == "daily":
    plot_data = output_calibration.copy()
elif plot_frequency == "monthly":
    plot_data = output_calibration.resample("M").agg({"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", "Qobs": "sum",  \
                                                      "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
                                                      "HBV_soil_moisture": "sum", "HBV_upper_gw": "sum", "HBV_lower_gw": "sum"})
elif plot_frequency == "yearly":
    plot_data = output_calibration.resample("Y").agg(
        {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", "Qobs": "sum", \
         "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
         "HBV_soil_moisture": "sum", "HBV_upper_gw": "sum", "HBV_lower_gw": "sum"})

stats = stats.create_statistics(output_calibration)
stats.to_csv(output_path + "model_stats_" +str(output_calibration.index.values[1])[:4]+"-"+str(output_calibration.index.values[-1])[:4]+".csv")

## Plotting the output data
# Plot the meteorological data
fig = plots.plot_meteo(plot_data)
if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "meteorological_data_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot the runoff data
fig1 = plots.plot_runoff(plot_data)
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
