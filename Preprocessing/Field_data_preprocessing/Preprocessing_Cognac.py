import pandas as pd
from pathlib import Path
import sys
import xarray as xr
home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/Preprocessing/ERA5_downscaling/')
from Preprocessing_functions import *
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'

data = xr.open_dataset(home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182ERA5_Land_1981_2019.nc')