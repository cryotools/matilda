import numpy as np
import xarray as xr
import pandas as pd
import salem
from pathlib import Path
import sys
import glob
home = str(Path.home())
# working_directory = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/'
working_directory = 'G:/CMIP5/'

def cmip_open(path, lat, lon, variable, method='nearest'):
    ds = xr.open_dataset(path)
    pick = ds.sel(lat=lat, lon=lon, method=method)
    dat = pick[[variable]].to_dataframe()[variable]
    return dat

cmip_open(working_directory + 'pr_day_EC-EARTH_rcp45_r7i1p1_20060101-20251231.nc', lat=41.0, lon=75.9, variable='pr')

ds = xr.open_dataset(working_directory + 'pr_day_EC-EARTH_rcp45_r7i1p1_20060101-20251231.nc')
pick = ds.sel(lat=41.0, lon=75.9, method='nearest')

variable = ['tas','pr']
scenario = ['rcp45', 'rcp85']
data_all = []
for scen in scenario:
    data_conc = []
    for var in variable:
        data_list = []
        for file in sorted(glob.glob(working_directory + var + '*' + scen + '*.nc')):
            data_list.append(cmip_open(file, lat=41.0, lon=75.9, variable=var))
        data_conc.append(pd.concat(data_list))
    data_all.append(pd.concat(data_conc, axis=1))

rcp_data = pd.concat(data_all, axis=1)
rcp_data.columns = ['temp_45', 'prec_45', 'temp_85', 'prec_85']
rcp_data.prec_45 = rcp_data.prec_45 * 86400 # prec in kg m-2 s-2
rcp_data.prec_85 = rcp_data.prec_85 * 86400 # prec in kg m-2 s-2

rcp_data.describe()

rcp_data.to_csv(working_directory + "temp_prec_rcp45_rcp85_2006-2050.csv")