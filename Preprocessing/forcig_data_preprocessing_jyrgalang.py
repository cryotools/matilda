##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import socket
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/matilda/Preprocessing'
import os
os.chdir(wd + '/Downscaling')
sys.path.append(wd)

##########################
#   Data preparation:    #
##########################

## ERA5L Gridpoint:

# Apply '/Ana-Lena_Phillip/data/matilda/Tools/ERA5_Subset_Routine.sh' for ncdf-subsetting