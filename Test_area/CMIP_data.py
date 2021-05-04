import numpy as np
import xarray as xr
import pandas as pd
import salem
from pathlib import Path
import sys
import glob
import subprocess
import os
home = str(Path.home())
working_directory = "/home/ana/Seafile/Tianshan_data/CMIP/CMIP6"

## output information
variable = ["tas", "pr"]
experiment = ['ssp1_2_6', 'ssp2_4_5']
model = ['awi_cm_1_1_mr', 'mpi_esm1_2_lr']
start_date = '2000-01-01'; end_date = '2019-12-31'
lat = 42.25; lon = 78.25

##
os.putenv('EXPERIMENT', ' '.join(experiment))
os.putenv('MODEL', ' '.join(model))

os.chdir(path="/home/ana/Desktop/")
subprocess.call(['./cmip_test3', working_directory, str(lon), str(lat)])


##
model = ["in_cm4_8"]
working_directory = "/home/ana/Desktop/"

def cmip_open(path, lat, lon, variable, method='nearest'):
    ds = xr.open_dataset(path)
    pick = ds.sel(lat=lat, lon=lon, method=method)
    dat = pick[[variable]].to_dataframe()
    time = ds.indexes['time'].to_datetimeindex()
    dat = dat.set_index(time)
    dat = dat[variable]
    return dat


data_all = []
columns = []
rcp_dfs = {}

for models in model:
    for scen in experiment:
        data_conc = []
        for var in variable:
            data_list = []
            for file in sorted(glob.glob(working_directory + models + '/' + var + '/' + scen + '/' + '*.nc')):
                data_list.append(cmip_open(file, lat=41.0, lon=75.9, variable=var))
            data_conc.append(pd.concat(data_list))
            data_name = [str(var) + '_' + str(scen)]
            columns = columns + data_name
        data_all.append(pd.concat(data_conc, axis=1))
        rcp_data = pd.concat(data_all, axis=1)
        rcp_data.columns = columns
        rcp_data = rcp_data[start_date:end_date]
    rcp_dfs[models] = rcp_data

## plots
import matplotlib.pyplot as plt

fig, ax = plt.subplots(len(rcp_dfs), 1, figsize=(7,5))
for a, key in zip(ax, rcp_dfs.keys()):
    df = rcp_dfs[key]
    a.plot(df.index.to_pydatetime(), df["tas_ssp1_2_6"])
    a.plot(df.index.to_pydatetime(), df["tas_ssp2_4_5"])
    # add labels/titles and such here

plt.show()