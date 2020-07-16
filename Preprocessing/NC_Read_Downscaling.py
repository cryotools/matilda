from pathlib import Path; home = str(Path.home())      ### Zieht sich home vom system
#import sys; sys.path.append(home + '/Seafile/Ana-Lena_Phillip/scripts/')
import xarray as xr
import pandas as pd
import numpy as np

# File organization
working_directory = '/Seafile/Ana-Lena_Phillip/data/input_output/'

era5_static_file = home + working_directory + 'ERA5/global/ERA5_land_Z_geopotential.nc'

era5_file_evap = home + working_directory + "/ERA5/202004_Urumqi_era5_evap_run_2011-2018.nc"
era5_file_met = home + working_directory + "ERA5/No1_Urumqi_ERA5_2000_201907.nc"

#Time slice
time_start = '2011-01-01T00:00'
time_end = '2018-12-31T23:00'

output = home + working_directory + 'input/202004_Umrumqi_ERA5_' + time_start.split('-')[0] + '_' + time_end.split('-')[0]   # Heisst input, weil es zwar hier der Output ist aber der COSIPY-Input.
output_cosipy = output + '_HBV.csv'

# Study Area Data
latitude_urumqi = 43.00; longitude_urumqi = 86.75                         ### Urumqi
target_altitude = 4025                                      # m a.s.l Urumqi
timezone_difference_to_UTC = 6

# Load data
era5_static = xr.open_dataset(era5_static_file)
era5_evap = xr.open_dataset(era5_file_evap)
era5_met = xr.open_dataset(era5_file_met)

# Downscaling Lapse rate pev calculation
pev_mean = -era5_evap.pev # PEV is inverted
pev_mean = pev_mean.mean(dim="time")
pev_mean = pev_mean.to_dataframe()
pev_mean.reset_index(inplace=True)

z_urumqi = era5_static.z.sel(lat=latitude_urumqi, lon=longitude_urumqi, method="nearest")
elevation_urumqi = z_urumqi.values/9.80665

longitude = era5_evap.longitude; latitude = era5_evap.latitude # Area for Pev Downscaling
era5_static = era5_static.sel(lat=latitude, lon=longitude, method="nearest")
era5_static = era5_static.z
era5_static = era5_static.to_dataframe()
era5_static.reset_index(inplace=True)
era5_static["elevation"] = era5_static["z"]/9.80665

df_downscaling = pd.merge(pev_mean, era5_static)
df_downscaling = df_downscaling[["latitude", "longitude", "pev", "elevation"]]
y = df_downscaling.pev.values
x = df_downscaling.elevation.values

#Linear regression
# number of observations/points
n = np.size(x)
# mean of x and y vector
m_x, m_y = np.mean(x), np.mean(y)
# calculating cross-deviation and deviation about x
SS_xy = np.sum(y * x) - n * m_y * m_x
SS_xx = np.sum(x * x) - n * m_x * m_x
# calculating regression coefficients
b_1 = SS_xy / SS_xx
b_0 = m_y - b_1 * m_x

# Downscaling
lapse_rate_pev = b_1
lapse_rate_temperature = -0.006             # K/m  temperature lapse rate
lapse_rate_total_precipitation = 0.00
height_diff = target_altitude - elevation_urumqi

era5_evap = era5_evap.sel(latitude=latitude_urumqi, longitude=longitude_urumqi, method="nearest")
era5_met = era5_met.sel(lat=latitude_urumqi, lon=longitude_urumqi, method="nearest")

era5_evap = era5_evap.sel(time=slice(time_start,time_end))
era5_met = era5_met.sel(time=slice(time_start,time_end))

time = era5_evap['time'].to_index()                              # era5['time'] ist das Gleiche wie era5.time
time_local = time+pd.Timedelta(hours=timezone_difference_to_UTC)

temperature = era5_met['t2m'].values
temperature = temperature + height_diff * lapse_rate_temperature

total_precipitation = era5_met['tp'].values.flatten()
total_precipitation = total_precipitation * 1000 + height_diff * lapse_rate_total_precipitation
# convert from m to mm

potential_evaporation = -era5_evap["pev"].values
# convert from m to mm
potential_evaporation = potential_evaporation + height_diff * lapse_rate_pev * 1000
potential_evaporation[potential_evaporation < 0] = 0

runoff = (era5_evap["ro"])
runoff = runoff * 1000
# convert from m to mm

# convert into csv
raw_data_csv = {'TIMESTAMP': time_local,
            'T2': temperature,
            'Runoff': runoff,
            'RRR': total_precipitation,
            'Pev': potential_evaporation,
            }
df = pd.DataFrame(raw_data_csv, columns = ['TIMESTAMP', 'T2',  'RRR', 'Runoff', 'Pev'])
df.to_csv(output_cosipy, index=False)
