## import of necessary packages
from pathlib import Path
import sys
import socket
import pandas as pd

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'

sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/MATILDA')

from MATILDA_slim import MATILDA


## Setting file paths and parameters
working_directory = home + "/EBA-CA/Azamat_AvH/workflow/data/Jyrgalang/"
input_path = home + "/EBA-CA/Azamat_AvH/workflow/data/Jyrgalang/obs/"

data_csv = "edit_t2m_jyrgalang_ERA5L_1982_2020.csv"
observation_data = "runoff_jyrgalang_2000_2021.csv"

output = working_directory + "output/" + data_csv[:15]

df 	= pd.read_csv(working_directory + 'obs/' + data_csv)
obs	= pd.read_csv(working_directory + 'obs/' + observation_data)

set_up_start='2000-01-01 00:00:00'
set_up_end='2002-12-31 23:00:00'
sim_start='2000-01-01 00:00:00'
sim_end='2005-11-20 23:00:00'

## Run MATILDA

MATILDA.MATILDA_simulation(df, obs, output=output, set_up_start=set_up_start, set_up_end=set_up_end,
                                             sim_start=sim_start, sim_end=sim_end, freq="D", lat=40,
                                             area_cat=254, area_glac=1.71, ele_dat=3864,
                                             ele_glac=3698.9, ele_cat=2970.6, TT_snow=0, TT_rain=2)
                                             



