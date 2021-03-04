import pandas as pd
from MATILDA_slim import MATILDA

## Model input
working_directory = "...MATILDA/"
data_csv = "data_2010-2019.csv"
obs_csv = "observations_2010_2019.csv"
output = working_directory + "Output/"

df = pd.read_csv(working_directory + data_csv)
obs = pd.read_csv(working_directory + obs_csv)

## Running the model step by step
# specifying all the catchment and model parameters
parameter = MATILDA.MATILDA_parameter(df, set_up_start='2010-01-01 00:00:00', set_up_end='2010-12-31 23:00:00',
                       sim_start='2011-01-01 00:00:00', sim_end='2019-11-01 23:00:00', freq="D", area_cat=50, area_glac=5,
                       ele_dat=3500, ele_glac=4000, ele_cat=3600)
# Data preprocessing
df, obs = MATILDA.MATILDA_preproc(df, parameter, obs=obs)

# running the submodules
output_MATILDA = MATILDA.MATILDA_submodules(df, parameter, obs=obs) # MATILDA model run + downscaling

# Creating plot for the input (meteorological) data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3) and
# adding them to the output
output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)

#saving the output
MATILDA.MATILDA_save_output(output_MATILDA, parameter, output)

## Running all the functions in one simulation
output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs, output=output, set_up_start='2010-01-01 00:00:00', set_up_end='2010-12-31 23:00:00',
                       sim_start='2011-01-01 00:00:00', sim_end='2019-11-01 23:00:00', freq="D", area_cat=50, area_glac=5,
                       ele_dat=3500, ele_glac=4000, ele_cat=3600)
