"""MATILDA Package - Example Script"""
import pandas as pd
from MATILDA_slim import MATILDA

## Model input
working_directory = ".../MATILDA/"
data_csv = "Data/forcing_data_2010-2020.csv"
obs_csv = "Data/runoff_2011-2020.csv"
output = working_directory + "Output/"

df = pd.read_csv(working_directory + data_csv)
obs = pd.read_csv(working_directory + obs_csv)

# Setting the parameter
# Specify the spin up and simulation period. One year spin up period is recommened
# Specify catchment properties like mean elevation, latitude and area
# Specify the individual model parameters, if none are specified, the standard parameters are used
parameter = MATILDA.MATILDA_parameter(df, set_up_start='2010-01-01 00:00:00', set_up_end='2010-12-31 23:00:00',
                                     sim_start='2011-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D", 						
                                      lat=40, area_cat=5.5, area_glac=2.2, ele_dat=3500, ele_glac=4000, 						
                                      ele_cat=3650, CFMAX_ice=5.5, CFMAX_snow=3)
# Data preprocessing
df, obs = MATILDA.MATILDA_preproc(df, parameter, obs=obs)

# running the submodules --> MATILDA model run + downscaling
# if you want to use the deltaH, include "glacier_profile =" and your glacier profile as a dataframe with Elevations (m), Area (fraction of the whole catchment) and WE (ice thickness in m w.e.) (see example glacier_profile) 
output_MATILDA = MATILDA.MATILDA_submodules(df, parameter, obs=obs)

# Creating plot for the input data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3)
output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)

#saving the output
MATILDA.MATILDA_save_output(output_MATILDA, parameter, output)

## Running all the functions in one simulation
output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs, output=output, set_up_start='2010-01-01 12:00:00', 							
                                            set_up_end='2010-12-31 12:00:00', sim_start='2011-01-01 12:00:00',
                                            sim_end='2020-12-31 12:00:00', freq="D", lat=40, area_cat=5.5, 							
                                            area_glac=2.2, ele_dat=3500, ele_glac=4000, ele_cat=3650, 
                                            CFMAX_ice=5.5, CFMAX_snow=3)
