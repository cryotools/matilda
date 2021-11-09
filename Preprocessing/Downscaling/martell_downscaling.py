##
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import salem
from pathlib import Path
import sys
import os
home = str(Path.home()) + '/Seafile'
wd = home + '/Martelltal/2021/MATILDA-Projekt'

in_file = wd + '/20211008_era5_t2_tp_2018-2021_martell.nc'
ds = xr.open_dataset(in_file)
pick = ds.sel(latitude=46.469395, longitude=10.673671, method='nearest')      # closest to Marteller Huette
era = pick.to_dataframe().filter(['t2m', 'tp'])

total_precipitation = np.append(0, (era.drop(columns='t2m').diff(axis=0).values.flatten()[1:]))   # transform from cumulative values
total_precipitation[total_precipitation < 0] = era.tp.values[total_precipitation < 0]
era['tp'] = total_precipitation

# era['tp'][era['tp'] < 0.00004] = 0         # Refer to https://confluence.ecmwf.int/display/UDOC/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ
era['tp'] = era['tp']*1000                 # Unit to mm

era_D = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})
era.to_csv(wd + '/t2m_tp_ERA5L_martell_lat46.5_lon10.7_20180101-20210801_hourly.csv')
era_D.to_csv(wd + '/t2m_tp_ERA5L_martell_lat46.5_lon10.7_20180101-20210801_daily.csv')


## Get reference elevation of selected ERA5L-gridcell:

z = wd + '/ERA5_land_Z_geopotential.nc'
ds = xr.open_dataset(z)
pick_z = ds.sel(lat=46.469395, lon=10.673671, method='nearest')
elev = pck_z.z.values/9.80665                             # (m/s^(-1)) Gravitational acceleration
print("Reference elevation of the selected grid cell is", round(elev, 2), "m.asl.")

## Read obs data

obs = pd.read_csv('/home/phillip/Seafile/Martelltal/2021/Minikin-MartellerHuette/'
                  'minikin2_marteller-huette_2021_09_03.csv', )