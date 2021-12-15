import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import socket
from bias_correction import BiasCorrection
import os

warnings.filterwarnings("ignore")  # sklearn
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/matilda/Preprocessing'
os.chdir(wd + '/Downscaling')
sys.path.append(wd)

##########################
#   Data preparation:    #
##########################

## ERA5L Gridpoint:

# Apply '/Ana-Lena_Phillip/data/matilda/Tools/ERA5_Subset_Routine.sh' for ncdf-subsetting

in_file = home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Jyrgalang/t2m_tp_jyrgalang_ERA5L_1982_2020.nc'
ds = xr.open_dataset(in_file)

pick = ds.sel(latitude=42.516, longitude=79.0167, method='nearest')           # closest to catchment center
# pick = pick.sel(time=slice('1989-01-01', '2019-12-31'))                      # start of gauging till end of file
era = pick.to_dataframe().filter(['t2m', 'tp'])

total_precipitation = np.append(0, (era.drop(columns='t2m').diff(axis=0).values.flatten()[1:]))   # transform from cumulative values
total_precipitation[total_precipitation < 0] = era.tp.values[total_precipitation < 0]
era['tp'] = total_precipitation

era['tp'][era['tp'] < 0.000004] = 0          # Refer to https://confluence.ecmwf.int/display/UDOC/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ
era['tp'] = era['tp']*1000

era_D = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})
era_D.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Jyrgalang/t2m_tp_jyrgalang_79.0-42.5_ERA5L_1982_2020.csv')


# Test for altitude of gridpoint
era5l_static = xr.open_dataset(home + "/Ana-Lena_Phillip/data/input_output/input/ERA5/global/ERA5_land_Z_geopotential.nc")

pick_z = era5l_static.sel(lat=42.516, lon=79.0167, method='nearest')
print("Reference altitude of gridpoint is", round(pick_z.z.values/9.80665, 2), "m.asl.")


## CMIP6:

cmip_path = home + '/Ana-Lena_Phillip/data/input_output/input/CMIP6/jyrgalang/'
cmip = pd.read_csv(cmip_path + 'CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31.csv',
                   index_col='time', parse_dates=['time'])

scen = ['26', '45', '70', '85']
cmip_edit = {}
for s in scen:
    cmip_scen = cmip.filter(like=s)
    cmip_scen.columns = ['t2m', 'tp']
    cmip_scen = cmip_scen.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})
    cmip_edit[s] = cmip_scen
    # cmip_scen.to_csv(cmip_path + 'CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31' + 'rcp' + s)


## Bias adjustment:

final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('1982-01-01', '2100-12-31')

# Temperature:
cmip_corrT = {}
for s in scen:
    x_train = cmip_edit[s][final_train_slice]['t2m'].squeeze()
    y_train = era_D[final_train_slice]['t2m'].squeeze()
    x_predict = cmip_edit[s][final_predict_slice]['t2m'].squeeze()
    bc_cmip = BiasCorrection(y_train, x_train, x_predict)
    cmip_corrT[s] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
    cmip_corrT[s].to_csv(cmip_path + 't2m_CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31_' + 'rcp' + s)

# Precipitation:
cmip_corrP = {}
for s in scen:
    x_train = cmip_edit[s][final_train_slice]['tp'].squeeze()
    y_train = era_D[final_train_slice]['tp'].squeeze()
    x_predict = cmip_edit[s][final_predict_slice]['tp'].squeeze()
    bc_cmip = BiasCorrection(y_train, x_train, x_predict)
    cmip_corrP[s] = pd.DataFrame(bc_cmip.correct(method='gamma_mapping'))
    cmip_corrP[s].to_csv(cmip_path + 'tp_CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31_' + 'rcp' + s)



# Plots:

t = slice('2017-01-01', '2020-12-31')

second = pd.DataFrame({'obs': era_D[t]['t2m'], 'mod': cmip_edit['26'][t]['t2m'], 'result': t_corr_cmip[t]['t2m']})
fig = plt.figure(1, figsize=(6, 4))
ax = fig.add_subplot(111)
bp = ax.boxplot(second)
ax.set_xticklabels(['era5_sdm (obs)', 'cmip6 (mod)', 'cmip6_sdm (result)'])
plt.show()

freq = 'M'
fig, ax = plt.subplots(figsize=(6, 4))
cmip_edit['26'][t]['t2m'].resample(freq).mean().plot(ax=ax, label='cmip6 (mod)', legend=True)
t_corr_cmip[t]['t2m'].resample(freq).mean().plot(ax=ax, label='cmip6_sdm (result)', legend=True)
era_D[t]['t2m'].resample(freq).mean().plot(label='era5_sdm (obs)', ax=ax, legend=True)
ax.set_title('Monthly air temperature in example period')

plt.show()


t = slice('2000-01-01', '2100-12-31')
freq = 'Y'
fig, ax = plt.subplots(figsize=(12, 8))
t_corr_cmip['t2m'][t].resample(freq).mean().plot(ax=ax, label='cmip_sdm (result)', legend=True)
cmip_edit['26'][t]['t2m'].resample(freq).mean().plot(label='cmip6 (scen)', ax=ax, legend=True)
era_D[t]['t2m'].resample(freq).mean().plot(label='era (obs)', ax=ax, legend=True)

plt.show()

