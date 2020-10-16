"""
This if the configuration file for the model framework. Adjust the paths, settings and parameters here
"""
from pathlib import Path; home = str(Path.home())
import os
from datetime import datetime

# Directories		# Defining home could be misleading. Better full paths?

working_directory = home + "/Seafile/Ana-Lena_Phillip/data/scripts/Final_Model/"

input_path_cosipy = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/best_cosipyrun_no1/best_cosipyrun_no1_2011-18/"
input_path_observations = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/glacierno1/hydro/"

	# Cirrus directories
#working_directory = "/data/projects/ebaca/data/scripts/centralasianwaterresources/Final_Model/"
#input_path_cosipy = "/data/projects/ebaca/data/input_output/input/best_cosipyrun_no1/best_cosipyrun_no1_2011-18/"
#input_path_observations = home + "/data/projects/ebaca/data/input_output/input/observations/glacierno1/hydro/"

input_path_data = home + ""
cosipy_nc = "best_cosipy_output_no1_2011-18.nc"
data_csv = "best_cosipy_input_no1_2011-18.csv"

output_path = working_directory + "Output/" + cosipy_nc[:-3] + ">>" + datetime.now().strftime("%Y-%m-%d_%H:%M:%S") + "/"
# output_path = working_directory + "Output/"
os.mkdir(output_path)

# dataframe with two columns, date and observations
observation_data = "daily_observations_2011-18.csv"
# Optional evapotranspiration dataframe, two columns with date and data
evap_data = " "

# Model configuration
area_name = "Urumqi" # specify the name of your catchment here
# Time period to calibrate initial parameters
cal_period_start = '2011-01-01 00:00:00' # beginning of the calibration period
cal_period_end = '2011-12-31 23:00:00' # end of calibration: one year is recommended
# Time period of the model simulation
sim_period_start = '2012-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2018-12-31 23:00:00'
cal_exclude = False # Excluding calibration period from statistics and plots
# Plot output
plot_frequency = "weekly" # Plot uses daily, weekly, monthly or yearly variables
plot_save = True # Plots are saved in the directory instead of being displayed only
compare_cosipy = True # compare model output with output from Cosipy

# Variables
# Temperature should be in Celsius
# Precipitation
prec_unit = True # Precipitation unit is in mm
prec_conversion = 1000 # Conversion factor through division
# Evapotranspiration: ET data is available, else it will be calculated using the formula by Oudin et al. (2005)
evap_unit = False # unit is mm
evap_conversion = 1000 # Conversion factor through division / day
# Runoff observation: unit is mm / day
