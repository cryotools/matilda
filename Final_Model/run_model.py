# -*- coding: UTF-8 -*-
"""
The model is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of the
glacierized catchments.
This file uses the input files created by the COSIPY-utility "aws2cosipy" as forcing data and additional
observation runoff data to validate it.
"""
## Running all the model functions
import sys
sys.path.append("/home/ana/Seafile/SHK/Scripts/centralasiawaterresources/Final_Model")
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import xarray as xr

from ConfigFile import *
from Scripts.DDM import calculate_PDD, calculate_glaciermelt
from Scripts.HBV import hbv_simulation
from Scripts.stats_plots import *

## Data input preprocessing
print('---')
print('Read input netcdf file %s' % (cosipy_nc))
print('Read input csv file %s' % (data_csv))
print('Read observation data %s' % (observation_data))
# Import necessary input: cosipy.nc, cosipy.csv and runoff observation data
# Observation data should be given in form of a csv with a date column and daily observations
ds = xr.open_dataset(input_path_cosipy + cosipy_nc)
df = pd.read_csv(input_path_cosipy + data_csv) # dataframe with temperature, precipitation and if possible evaporation
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
degreedays_ds = calculate_PDD(ds)
# Calculating runoff and melt
output_DDM = calculate_glaciermelt(degreedays_ds) # output in mm
print(output_DDM.head(5))

## HBV model
print("Running the HBV model")
# Runoff calculations for the catchment with the HBV model
output_hbv = hbv_simulation(df, cal_period_start, cal_period_end) # output in mm
print(output_hbv.head(5))

## Output postprocessing
output = pd.concat([output_hbv, output_DDM], axis=1)
output = pd.concat([output, obs], axis=1)
output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]

nash_sut = NS(output["Qobs"], output["Q_Total"]) # Nash–Sutcliffe model efficiency coefficient
print("The Nash–Sutcliffe model efficiency coefficient of the total model is " + str(nash_sut))

print("Writing the output csv to disc")
output_csv = output.copy()
output_csv = output_csv.fillna(0)
output_csv.to_csv(output_path + "model_output_" +str(cal_period_start[:4])+"-"+str(sim_period_end[:4]+".csv"))

## Statistical analysis
# Calibration period
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
                                                      "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})
elif plot_frequency == "yearly":
    plot_data = output_calibration.resample("Y").agg(
        {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", "Qobs": "sum", \
         "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
         "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})

stats = create_statistics(output_calibration)
stats.to_csv(output_path + "model_stats_" +str(output_calibration.index.values[1])[:4]+"-"+str(output_calibration.index.values[-1])[:4]+".csv")

# Including Cosipy in the evaluation
if compare_cosipy == True:# Including Cosipy in the evaluation
    output_cosipy = output[{"Qobs", "Q_Total", "DDM_smb", "DDM_total_melt"}]
    cosipy_runoff = ds.Q.mean(dim=["lat", "lon"])
    cosipy_smb = ds.surfMB.mean(dim=["lat", "lon"])
    #cosipy_smb = cosipy_smb.resample(time="D").sum(dim="time")
    cosipy_melt = ds.surfM.mean(dim=["lat", "lon"])
    #cosipy_melt = cosipy_melt.resample(time="D").sum(dim="time")
    output_cosipy["Q_COSIPY"] = cosipy_runoff.resample(time="D").sum(dim="time")*1000
    output_cosipy["COSIPY_smb"] = cosipy_smb.to_dataframe().surfMB.resample('D').sum()*1000
    output_cosipy["COSIPY_melt"] = cosipy_melt.to_dataframe().surfM.resample('D').mean()*1000
    nash_sut_cosipy = NS(output_cosipy["Qobs"], output_cosipy["Q_COSIPY"])
    output_cosipy.to_csv(output_path + "cosipy_comparison_output_" + str(cal_period_start[:4]) + "-" + str(sim_period_end[:4] + ".csv"))

    stats_cosipy = create_statistics(output_cosipy)
    stats_cosipy.to_csv(output_path + "cosipy_comparison_stats_" + str(output_calibration.index.values[1])[:4] + "-" + str(
        output_calibration.index.values[-1])[:4] + ".csv")
    if plot_frequency == "daily":
        plot_data_cosipy = output_cosipy.copy()
    elif plot_frequency == "monthly":
        plot_data_cosipy = output_cosipy.resample("M").agg(
            {"Qobs": "sum", "Q_Total": "sum", "Q_COSIPY": "sum", "DDM_smb": "sum", "DDM_total_melt": "sum", \
             "COSIPY_smb": "sum", "COSIPY_melt": "sum"})
    elif plot_frequency == "yearly":
        plot_data_cosipy = output_cosipy.resample("Y").agg(
            {"Qobs": "sum", "Q_Total": "sum", "Q_COSIPY": "sum", "DDM_smb": "sum", "DDM_total_melt": "sum", \
             "COSIPY_smb": "sum", "COSIPY_melt": "sum"})

## Plotting the output data
# Plot the meteorological data
fig = plot_meteo(plot_data)
if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "meteorological_data_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot the runoff data
fig1 = plot_runoff(plot_data, nash_sut)
if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "model_runoff_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot the HBV paramters
fig2 = plot_hbv(plot_data)
if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "HBV_output_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

if compare_cosipy == True:
    fig3 = plot_cosipy(plot_data_cosipy, nash_sut, nash_sut_cosipy)
    if plot_save == False:
        plt.show()
    else:
        plt.savefig(
            output_path + "COSIPY_output_" + str(plot_data.index.values[1])[:4] + "-" + str(plot_data.index.values[-1])[
                                                                                     :4] + ".png")

print('Saved plots of meteorological and runoff data to disc')
print("End of model run")
print('---')

##

