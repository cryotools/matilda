##
from pathlib import Path; home = str(Path.home())
import os
import pandas as pd
import numpy as np
import xarray as xr

def create_statistics(df):
    stats = df.describe()
    sum = pd.DataFrame(df.sum())
    sum.columns = ["sum"]
    sum = sum.transpose()
    stats = stats.append(sum)
    stats = stats.round(2)
    return stats

year ## Data
ba_aws = pd.read_csv(home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/bashkaingdy/met/obs/aws_preprocessed_2017-06_2021-05.csv", index_col='time', parse_dates=['time'])
ba_era = pd.read_csv(home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/t2m_tp_ERA5L_no182_41.1_75.9_1982_2019.csv', index_col='time', parse_dates=['time'])
ba_era_down = pd.read_csv(home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/bashkaingdy/met/era5l/no182_ERA5_Land_1982_2020_41_75.9_fitted2AWS.csv", index_col='time', parse_dates=['time'])
ba_cmip_down = pd.read_csv(home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/bashkaingdy/met/cmip6/CMIP6_mean_41-75.9_1980-01-01-2100-12-31_downscaled.csv", index_col='time', parse_dates=['time'])
ba_cmip = pd.read_csv(home + '/Seafile/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/CMIP6_mean_41-75.9_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])


ky_aws = pd.read_csv(home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/obs/met_data_full_kyzylsuu_2007-2015.csv", index_col='time', parse_dates=['time'])
ky_era = pd.read_csv(home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/t2m_tp_ERA5L_kyzylsuu_42.2_78.2_1982_2019.csv', index_col='time', parse_dates=['time'])
ky_era_down = pd.read_csv(home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/era5l/kyzylsuu_ERA5_Land_1982_2020_42.2_78.2_fitted2AWS.csv", index_col='time', parse_dates=['time'])
ky_cmip_down = pd.read_csv(home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/cmip6/CMIP6_mean_42.25-78.25_1980-01-01-2100-12-31_downscaled.csv", index_col='time', parse_dates=['time'])
ky_cmip = pd.read_csv(home + '/Seafile/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/CMIP6_mean_42.25-78.25_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])

## ERA to AWS
final_train_slice = slice('2017-07-14', '2020-12-31')
final_predict_slice = slice('1982-01-01', '2020-12-31')

ba_aws["t2m"] = ba_aws["t2m"] - 273.15
ba_era["t2m"] = ba_era["t2m"] - 273.15
ba_era_down["t2m"] = ba_era_down["t2m"] - 273.15


ba_aws = ba_aws[final_train_slice].resample('Y').agg({'t2m': 'mean', 'tp': 'sum'}) #AWS Training
ba_era_training = ba_era[final_train_slice].resample('Y').agg({'t2m': 'mean', 'tp': 'sum'}) # ERA5 training
ba_era_full = ba_era[final_predict_slice].resample('Y').agg({'t2m': 'mean', 'tp': 'sum'}) # ERA5 full period

ba_era_down_full = ba_era_down[final_predict_slice].resample('Y').agg({'t2m': 'mean', 'tp': 'sum'}) # ERA5 Downscaled full period
print("Bash Kaindy AWS Training" + os.linesep + str(create_statistics(ba_aws)))
print("Bash Kaindy ERA Training" + os.linesep + str(create_statistics(ba_era_training)))
print("Bash Kaindy full ERA" + os.linesep + str(create_statistics(ba_era_full)))
print("Bash Kaindy full ERA downscaled" + os.linesep + str(create_statistics(ba_era_down_full)))


## CMIP to ERA
final_predict_slice = slice('2000-01-01', '2100-12-30')
for y in ba_cmip.loc[:, ba_cmip.columns.str.startswith('temp')].columns:
    ba_cmip[y] = ba_cmip[y] - 273.15

for y in ba_cmip_down.loc[:, ba_cmip_down.columns.str.startswith('temp')].columns:
    ba_cmip_down[y] = ba_cmip_down[y] - 273.15

ba_cmip = ba_cmip[final_predict_slice].resample('Y').agg({'temp_26': 'mean','temp_45': 'mean','temp_85': 'mean', 'prec_26': 'sum', 'prec_45': 'sum', 'prec_85': 'sum'})
ba_cmip_down = ba_cmip_down[final_predict_slice].resample('Y').agg({'temp_26': 'mean','temp_45': 'mean','temp_85': 'mean', 'prec_26': 'sum', 'prec_45': 'sum', 'prec_85': 'sum'})

print("Bash Kaindy CMIP" + os.linesep + str(create_statistics(ba_cmip)))
print("Bash Kaindy CMIP downscaled" + os.linesep + str(create_statistics(ba_cmip_down)))

## ERA to AWS:
final_train_slice = slice('2007-08-10', '2016-01-01')
final_predict_slice = slice('1982-01-01', '2020-12-31')

ky_aws["t2m"] = ky_aws["t2m"] - 273.15
ky_era["t2m"] = ky_era["t2m"] - 273.15
ky_era_down["t2m"] = ky_era_down["t2m"] - 273.15

ky_aws = ky_aws[final_train_slice].resample('Y').agg({'t2m': 'mean', 'tp': 'sum'}) #AWS Training
ky_era_training = ky_era[final_train_slice].resample('Y').agg({'t2m': 'mean', 'tp': 'sum'}) # ERA5 training
ky_era_full = ky_era[final_predict_slice].resample('Y').agg({'t2m': 'mean', 'tp': 'sum'}) # ERA5 full period

ky_era_down_full = ky_era_down[final_predict_slice].resample('Y').agg({'t2m': 'mean', 'tp': 'sum'}) # ERA5 Downscaled full period
print("Kyzylsuu AWS Training" + os.linesep + str(create_statistics(ky_aws)))
print("Kyzylsuu ERA Training" + os.linesep + str(create_statistics(ky_era_training)))
print("Kyzylsuu full ERA" + os.linesep + str(create_statistics(ky_era_full)))
print("Kyzylsuu full ERA downscaled" + os.linesep + str(create_statistics(ky_era_down_full)))
## CMIP to ERA:
final_train_slice = slice('1982-01-01', '2019-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-31')

for y in ky_cmip.loc[:, ky_cmip.columns.str.startswith('temp')].columns:
    ky_cmip[y] = ky_cmip[y] - 273.15

for y in ky_cmip_down.loc[:, ky_cmip_down.columns.str.startswith('temp')].columns:
    ky_cmip_down[y] = ky_cmip_down[y] - 273.15

ky_cmip = ky_cmip[final_predict_slice].resample('Y').agg({'temp_26': 'mean','temp_45': 'mean','temp_85': 'mean', 'prec_26': 'sum', 'prec_45': 'sum', 'prec_85': 'sum'})
ky_cmip_down = ky_cmip_down[final_predict_slice].resample('Y').agg({'temp_26': 'mean','temp_45': 'mean','temp_85': 'mean', 'prec_26': 'sum', 'prec_45': 'sum', 'prec_85': 'sum'})


print("Kyzylsuu CMIP" + os.linesep + str(create_statistics(ky_cmip)))
print("Kyzylsuu CMIP downscaled" + os.linesep + str(create_statistics(ky_cmip_down)))

## Test for altitude of gridpoint ERA5L
era5l_static = xr.open_dataset(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/global/ERA5_land_Z_geopotential.nc")
era5_static = xr.open_dataset(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/global/ERA5_global_z.nc")


ds = xr.open_dataset(home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/t2m_tp_kysylsuu_ERA5L_1982_2020.nc')
pick_ky = ds.sel(latitude=42.191433, longitude=78.200253, method='nearest')

ds = xr.open_dataset(home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/new_grib_conversion' +\
          '/t2m_tp_no182_ERA5L_1982_2020.nc')
pick_ba = ds.sel(latitude=41.134066, longitude=75.942381, method='nearest')

era5l_static_ba = era5l_static.sel(lat=41.1, lon=75.9, method='nearest')
era5l_static_ky = era5l_static.sel(lat=42.2, lon=78.2, method='nearest')
era5_static_ba = era5_static.sel(lat=41.1, lon=75.9, method='nearest')
era5_static_ky = era5_static.sel(lat=42.2, lon=78.2, method='nearest')

print("Altitude of gridpoint BA", era5l_static_ba.z.values/9.80665)
print("Altitude of gridpoint KY", era5l_static_ky.z.values/9.80665)
print("Altitude of gridpoint BA (z from ERA5)", era5_static_ba.z.values/9.80665)
print("Altitude of gridpoint KY (z from ERA5)", era5_static_ky.z.values/9.80665)



