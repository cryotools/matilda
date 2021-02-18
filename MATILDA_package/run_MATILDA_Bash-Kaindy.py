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
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/"
input_path_data = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/"
input_path_observations = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"

cosipy_nc = ""
data_csv = "no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
observation_data = "runoff_bashkaindy_04_2019-11_2020_test.csv" # Daily Runoff Observations in mm

# Additional information
# Warming up period to get appropriate initial values for various variables (e.g. GW level etc.)
set_up_start = '2018-01-01 00:00:00' # beginning of  period
set_up_end = '2019-12-31 23:00:00' # end of period: one year is recommended
# Time period of the model simulation
sim_period_start = '2019-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2020-11-01 23:00:00'

# output
output_path = working_directory + "input_output/output/" + data_csv[:15] + sim_period_start[:4] + "_" + sim_period_end[:4] + "_" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "/"
os.mkdir(output_path) # creates new folder for each model run with timestamp

glacier_area = 2.566
catchment_area = 46.232

# Downscaling the temperature and precipitation to glacier altitude for the DDM
lapse_rate_temperature = -0.006 # K/m
lapse_rate_precipitation = 0
height_diff_catchment = -504 # height data is 3864 m, catchment mean is 3360 glacier mean is 4042m
height_diff_glacier = 178

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

print("Set up period between " + str(set_up_start) + " and "  + str(set_up_end) + " to get appropriate initial values")
print("Simulation period between " + str(sim_period_start) + " and "  + str(sim_period_end))
df = dataformatting.data_preproc(df, set_up_start, sim_period_end) # formatting the input to right format
#ds = dataformatting.data_preproc(ds, set_up_start, sim_period_end)
obs = dataformatting.data_preproc(obs, set_up_start, sim_period_end)
#obs = obs.tz_localize('Asia/Bishkek')


# Downscaling the dataframe to the glacier height
df_DDM = dataformatting.glacier_downscaling(df, height_diff=height_diff_glacier, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)
df = dataformatting.glacier_downscaling(df, height_diff=height_diff_catchment, lapse_rate_temperature=lapse_rate_temperature, lapse_rate_precipitation=lapse_rate_precipitation)

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
output_DDM, parameter_DDM = DDM.calculate_glaciermelt(degreedays_ds, pdd_factor_snow=2.5, pdd_factor_ice=5, temp_snow=-0.5) # output in mm, parameter adjustment possible
output_DDM["Q_DDM"] = output_DDM["Q_DDM"]*(glacier_area/catchment_area) # scaling glacier melt to glacier area
# Calculating the mass balance
output_DDM["water_year"] = np.where((output_DDM.index.month) >= 9, output_DDM.index.year +1, output_DDM.index.year)
yearly_smb = output_DDM.groupby("water_year")["DDM_smb"].sum()/1000*0.9
print("The yearly mass balance (period from September to August)")
print(pd.DataFrame(yearly_smb))
print("Finished running the DDM")

## HBV model
print("Running the HBV model")
# Runoff calculations for the catchment with the HBV model
output_hbv, parameter_HBV = HBV.hbv_simulation(df, set_up_start, set_up_end, parTT=-0.5, parCFMAX=2.5, parPERC=2.5, parFC=200, parUZL=60, parMAXBAS=2) # output in mm, individual parameters can be set here
print("Finished running the HBV")
## Output postprocessing
#output_hbv["Q_HBV"] = output_hbv["Q_HBV"] - (output_DDM["DDM_snow_melt_rate"]*(glacier_area/catchment_area))
output = dataformatting.output_postproc(output_hbv, output_DDM, obs)
output = output[sim_period_start: sim_period_end]

nash_sut = stats.NS(output["Qobs"], output["Q_Total"]) # Nash–Sutcliffe model efficiency coefficient
if nash_sut == "error":
    print("ERROR. The Nash–Sutcliffe model efficiency coefficient is outside the range of -1 to 1")
else:
    print("The Nash–Sutcliffe model efficiency coefficient of the MATILDA run is " + str(round(nash_sut, 2)))


print("Writing the output csv to disc")
output_csv = output.copy()
output_csv = output_csv.fillna(0)
output_csv.to_csv(output_path + "model_output_" + str(output_csv.index.values[1])[:4]+"-"+str(output_csv.index.values[-1])[:4]+".csv")
parameter = dataformatting.output_parameter(parameter_HBV, parameter_DDM)
parameter.to_csv(output_path + "model_parameter.csv")
## Statistical analysis
# Daily, weekly, monthly or yearly output
plot_data = dataformatting.plot_data(output, plot_frequency)

stats_output = stats.create_statistics(output)
stats_output.to_csv(output_path + "model_stats_" +str(output.index.values[1])[:4]+"-"+str(output.index.values[-1])[:4]+".csv")
print("Output overview")
print(stats_output[["T2", "RRR", "PE", "Q_DDM", "Q_HBV", "Qobs", "Q_Total"]])
print("Yearly MB in 2019 " + str(round(smb_2019,2)))
print("Yearly MB in 2020 " + str(round(smb_2020,2)))
## Cosipy comparison
if cosipy == True:
    output_cosipy = dataformatting.output_cosipy(output, ds)
    output_cosipy.to_csv(output_path + "cosipy_comparison_output_" + str(set_up_start[:4]) + "-" + str(sim_period_end[:4] + ".csv"))

    nash_sut_cosipy = stats.NS(output_cosipy["Qobs"], output_cosipy["Q_COSIPY"])

    stats_cosipy = stats.create_statistics(output_cosipy)
    stats_cosipy.to_csv(output_path + "cosipy_comparison_stats_" + str(output.index.values[1])[:4] + "-" + str(
        output.index.values[-1])[:4] + ".csv")
    plot_data_cosipy = dataformatting.plot_data_cosipy(output_cosipy, plot_frequency, set_up_start, sim_period_end)

    fig3 = plots.plot_cosipy(plot_data_cosipy, plot_frequency_long, nash_sut, nash_sut_cosipy)
    if plot_save == False:
        plt.show()
    else:
        if str(plot_data_cosipy.index.values[1])[:4] == str(plot_data_cosipy.index.values[-1])[:4]:
            plt.savefig(output_path + "COSIPY_output_" + + str(plot_data_cosipy.index.values[-1])[:4] + ".png", dpi=fig3.dpi)
        else:
            plt.savefig(output_path + "COSIPY_output_" + str(plot_data_cosipy.index.values[1])[:4] + "-" + str(
            plot_data_cosipy.index.values[-1])[:4] + ".png", dpi=fig3.dpi)

## Plotting the output data
# Plot the meteorological data
fig = plots.plot_meteo(plot_data, plot_frequency_long)
if plot_save == False:
	plt.show()
else:
    if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
        plt.savefig(output_path + "meteorological_data_" + str(plot_data.index.values[-1])[:4]+".png", dpi=fig.dpi)
    else:
	    plt.savefig(output_path + "meteorological_data_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png", dpi=fig.dpi)

# Plot the runoff data
fig1 = plots.plot_runoff(plot_data, plot_frequency_long, nash_sut)
if plot_save == False:
	plt.show()
else:
    if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
        fig1.savefig(output_path + "model_runoff_" + str(plot_data.index.values[-1])[:4]+".png", dpi=fig1.dpi)
    else:
	    fig1.savefig(output_path + "model_runoff_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png", dpi=fig1.dpi)

# Plot the HBV paramters
fig2 = plots.plot_hbv(plot_data, plot_frequency_long)
if plot_save == False:
	plt.show()
else:
    if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
        plt.savefig(output_path + "HBV_output_" + str(plot_data.index.values[-1])[:4]+".png", dpi=fig.dpi)
    else:
	    plt.savefig(output_path + "HBV_output_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png", dpi=fig.dpi)

print('Saved plots of meteorological and runoff data to disc')
print("End of model run")
print('---')

