"""MATILDA Package - Example Script:
    Demonstrates the MATILDA workflow using a 3y example dataset. By default, the script reads the input files from
    the directory in which the script is located.
    The model is run with the comprehensive matilda_simulation function, and output is saved in the working directory.
"""
import os
import sys
import pandas as pd
from matilda.core import matilda_simulation

## Model input
working_directory = sys.path[0]      # Points to the folder where the script is located. Change to your needs.
os.chdir(working_directory)
df = pd.read_csv('forcing_data.csv')
obs = pd.read_csv('runoff_data.csv')

## Quick model run:
output_matilda = matilda_simulation(df, obs=obs, output=working_directory,
                                    set_up_start='2010-01-01', set_up_end='2010-12-31',  # Min. 1y recommended
                                    sim_start='2011-01-01', sim_end='2013-12-31',
                                    freq="D",  # Temporal resolution ("D", "M" or "Y")
                                    lat=42,  # Latitude
                                    area_cat=316,  # Catchment area
                                    area_glac=33,  # Glaciated area in the catchment
                                    ele_dat=2550,  # Reference altitude of the data (e.g. AWS altitude)
                                    ele_glac=4000,  # Mean altitude of glaciated area
                                    ele_cat=3650,  # Mean catchment elevation

                                    # Optional:
                                    # 1. specify model parameters, e.g.
                                    PCORR=1.5,
                                    # For a list of model parameters and default values check the Parameters file

                                    # 2. Add glacier profile (see Readme) to account for glacier change:
                                    # glacier_profile='glacier_profile.csv'
                                    elev_rescaling=False,

                                    # 3. Include interactive plots:
                                    plot_type='all'	# If you receive errors relating to plotly either try a different plotly version or change plot_type to "print"
                                    )



