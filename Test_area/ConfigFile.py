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
#os.mkdir(output_path)

# dataframe with two columns, date and observations
observation_data = "daily_observations_2011-18.csv"
# Optional evapotranspiration dataframe, two columns with date and data
evap_data = " "

# Model configuration
area_name = "Urumqi" # specify the name of your catchment here
# Time period to calibrate initial parameters
cal_period_start = '2011-01-01 00:00:00' # beginning of the calibration period
cal_period_end = '2013-12-31 23:00:00' # end of calibration: one year is recommended
# Time period of the model simulation
sim_period_start = '2014-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2018-12-31 23:00:00'
cal_exclude = False # Excluding calibration period from statistics and plots
# Plot output
plot_frequency = "monthly" # Plot uses daily, monthly or yearly variables
plot_save = True # Plots are saved in the directory instead of being displayed only
compare_cosipy = True # compare model output with output from Cosipy

# Variables
# Temperature
temp_unit = True # Temperature unit is Kelvin
# Precipitation
prec_unit = True # Precipitation unit is in mm
prec_conversion = 1000 # Conversion factor through division
# Evapotranspiration: unit is mm / day
evap_data_available = False # ET data is available, else it will be calculated using the formula by Oudin et al. (2005)
# Runoff observation: unit is mm / day

# Parameters for the DDM
"""
    *pdd_factor_snow* : float
        Positive degree-day factor for snow.
    *pdd_factor_ice* : float
        Positive degree-day factor for ice.
    *refreeze_snow* : float
        Refreezing fraction of melted snow.
    *refreeze_ice* : float
        Refreezing fraction of melted ice.
    *temp_snow* : float
        Temperature at which all precipitation falls as snow.
    *temp_rain* : float
        Temperature at which all precipitation falls as rain.
"""
parameters_DDM = {
    'pdd_factor_snow':  2.8, # according to Huintjes et al. 2010  [5.7 mm per day per Celsius according to Hock 2003]
    'pdd_factor_ice':   5.6, # according to Huintjes et al. 2010 [7.4 mm per day per Celsius according to Hock 2003]
    'temp_snow':        0.0,
    'temp_rain':        2.0,
    'refreeze_snow':    0.0,
    'refreeze_ice':     0.0}

# Parameters for the HBV model
"""
List of 16 HBV model parameters
    [parBETA, parCET,  parFC,    parK0,
    parK1,    parK2,   parLP,    parMAXBAS,
    parPERC,  parUZL,  parPCORR, parTT,
    parCFMAX, parSFCF, parCFR,   parCWH]

    # 16 PARAMETERS_HBV
    # BETA   - parameter that determines the relative contribution to runoff from rain or snowmelt
    #          [1, 6]
    # CET    - Evaporation correction factor
    #          (should be 0 if we don't want to change (Oudin et al., 2005) formula values)
    #          [0, 0.3]
    # FC     - maximum soil moisture storage
    #          [50, 500]
    # K0     - recession coefficient for surface soil box (upper part of SUZ)
    #          [0.01, 0.4]
    # K1     - recession coefficient for upper groudwater box (main part of SUZ)
    #          [0.01, 0.4]
    # K2     - recession coefficient for lower groudwater box (whole SLZ)
    #          [0.001, 0.15]
    # LP     - Threshold for reduction of evaporation (SM/FC)
    #          [0.3, 1]
    # MAXBAS - routing parameter, order of Butterworth filter
    #          [1, 7]
    # PERC   - percolation from soil to upper groundwater box
    #          [0, 3]
    # UZL    - threshold parameter for groundwater boxes runoff (mm)
    #          [0, 500]
    # PCORR  - Precipitation (input sum) correction factor
    #          [0.5, 2]
    # TT     - Temperature which separate rain and snow fraction of precipitation
    #          [-1.5, 2.5]
    # CFMAX  - Snow melting rate (mm/day per Celsius degree)
    #          [1, 10]
    # SFCF   - SnowFall Correction Factor
    #          [0.4, 1]
    # CFR    - Refreezing coefficient
    #          [0, 0.1] (usually 0.05)
    # CWH    - Fraction (portion) of meltwater and rainfall which retain in snowpack (water holding capacity)
    #          [0, 0.2] (usually 0.1)

    Output:
    simulated river runoff (daily timesteps)
"""

parameters_HBV=[ 1.0,   0.15,     250,   0.055, 0.055,   0.04,     0.7,     3.0,\
        1.5,    120,     1.0,     0.0, 5.0,    0.7,     0.05,    0.1]

parBETA, parCET, parFC, parK0, parK1, parK2, parLP, parMAXBAS,\
    parPERC, parUZL, parPCORR, parTT, parCFMAX, parSFCF, parCFR, parCWH = parameters_HBV
