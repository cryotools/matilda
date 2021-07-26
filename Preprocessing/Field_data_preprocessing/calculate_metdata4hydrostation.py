import numpy as np; import pandas as pd; import xarray as xr; import salem
from pathlib import Path
import matplotlib.pyplot as plt
import sys; sys.path.append('/home/phillip/Seafile/Ana-Lena_Phillip/data/matilda/Preprocessing/ERA5_downscaling/')
from fundamental_physical_constants import g, M, R

home = str(Path.home())
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'

##
time_start = '2018-09-07 18:00:00'  # longest timeseries of waterlevel
time_end = '2019-09-14 09:00:00'

aws_up = pd.read_csv(working_directory + 'Minikin/Cognac_glacier/80_2019_09_12_refin.csv', sep=';', decimal=',',
                     usecols=range(0, 4))
aws_up.columns = ['datetime', 'G', 'temp', 'hum']
aws_up.datetime = pd.to_datetime(aws_up.datetime)
aws_up.set_index(aws_up.datetime, inplace=True)
aws_up = aws_up.drop(['datetime'], axis=1)
aws_up = aws_up.shift(-2, axis=0)  # Only -2h timeshift results in the correct curves. BUT WHY????
aws_up.temp = aws_up.temp + 273.15
aws_up = aws_up[time_start: time_end]

aws_down = pd.read_csv(working_directory + 'AWS_atbs/temp_2017-05-30_2020-09-16.csv')
aws_down.columns = ['datetime', 'temp']
aws_down.datetime = pd.to_datetime(aws_down.datetime)
aws_down.set_index(aws_down.datetime, inplace=True)
aws_down = aws_down.drop(['datetime'], axis=1)
aws_down.temp = aws_down.temp + 273.15
aws_down = aws_down.resample('H').mean()
aws_down = aws_down[time_start: time_end]
#
# ##
# aws_far = pd.read_csv(working_directory + 'AWS_asai/press_hPa_2017-08-01_2020-09-15_cut.csv')
# aws_far.columns = ['datetime', 'air_press']
# aws_far.datetime = pd.to_datetime(aws_far.datetime)
# aws_far.set_index(aws_far.datetime, inplace=True)
# aws_far = aws_far.drop(['datetime'], axis=1)
# aws_far = aws_far.resample('H').mean()
#
# aws_far_temp = pd.read_csv(working_directory + 'AWS_asai/temp_2017-08-01_2020-09-15_cut.csv')
# aws_far_temp.columns = ['datetime', 'temp']
# aws_far_temp.datetime = pd.to_datetime(aws_far_temp.datetime)
# aws_far_temp.set_index(aws_far_temp.datetime, inplace=True)
# aws_far_temp = aws_far_temp.drop(['datetime'], axis=1)
# aws_far_temp.temp = aws_far_temp.temp + 273.15
# aws_far_temp = aws_far_temp.resample('H').mean()
#
# aws_far['temp'] = aws_far_temp.temp
# aws_far = aws_far[time_start: time_end]
# aws_far.air_press = aws_far.air_press[aws_far.air_press.between(680, 720)]      # Removes extreme outliers that influence the runoff curve.

##
alt_hobo = 3342
lat_hobo = 41.068228
lon_hobo = 75.99092
alt_aws_far = 3023
alt_aws_up = 3894
alt_aws_down = 2250
lapseT = -(aws_down.temp.mean() - aws_up.temp.mean()) / (alt_aws_down - alt_aws_up)  # literature: -0.006 K/m

##
era5_land_static_file = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/global/ERA5_land_Z_geopotential.nc'
era5_land_file = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/20200916_era5_t2_sp_2017-2019.nc'
target_altitude = alt_hobo
margin = 0.2
lon_ll = lon_hobo - margin; lat_ll = lat_hobo - margin
lon_ur = lon_hobo + margin; lat_ur = lat_hobo + margin
era5_land_static = salem.open_xr_dataset(era5_land_static_file)
era5_land_static = era5_land_static.salem.subset(corners=((lon_ll, lat_ll), (lon_ur, lat_ur)), crs=salem.wgs84)
era5_land = salem.open_xr_dataset(era5_land_file)
era5_land = era5_land.sel(time=slice(time_start, time_end))
altitude_differences_gp = np.abs(era5_land_static.z/g - target_altitude)
# latitude = float(era5_land_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lat)        # Doesn't work unfortunately...
# longitude = float(era5_land_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lon)

test = altitude_differences_gp == np.nanmin(altitude_differences_gp) # Check manually instead: most similar altitude at lat 40.9, lon 76.1
latitude = 40.9
longitude = 76.1

era5_land = era5_land.sel(latitude=latitude, longitude=longitude, method='nearest'); era5_land_static = era5_land_static.sel(lat=latitude,lon=longitude, method='nearest')      # called latitude/longitude in era5_land and lat/lon inera5_land_static
height_diff = target_altitude - era5_land_static.z.values/g; print("Height difference between target_altitude: ", height_diff)
print('First timestamp: ', era5_land.time[0].values, ' last timestamp: ', era5_land.time[-1].values)
print("Altitude of gridpoint ", era5_land_static.z.values/g)
air_pressure = ((era5_land['sp'].values)/100) * (1-abs(lapseT)*height_diff/era5_land['t2m'].values) ** \
               ((g*M)/(R*abs(lapseT)))
temperature = (era5_land['t2m'].values) + height_diff * lapseT
##
# temp_hobo = aws_down.temp + (
#             alt_hobo - alt_aws_down) * lapseT  # Calculate temperature at water level sensor location from aws_down series.
# temp_mean = (temp_hobo + aws_far.temp) / 2
#
#
# def p(p0, h, t0, lapseR):
#     return p0 * (1 - (lapseR * h) / t0) ** 5.255  # Calculate air pressure at altitude.
#
#
# p_hobo = p(aws_far.air_press, alt_hobo - alt_aws_far, temp_mean,
#            (aws_far.temp.mean() - aws_up.temp.mean()) / (alt_aws_far - alt_aws_up))
# p_hobo.describe()

##
data_hobo = round(pd.DataFrame({'datetime': aws_down.index, 'temp': temperature, 'press': air_pressure}), 2)
data_hobo.to_csv(working_directory + "HOBO_water/temp_press_hydrostation_2018-2019.csv", index=False)

##
# time_start = '2018-09-07 12:00:00'
# time_end = '2018-09-15 12:00:00'
# aws_up = aws_up[time_start: time_end]
# aws_down = aws_down[time_start: time_end]
# aws_far = aws_far[time_start: time_end]
# aws_comb = aws_down
# aws_comb['temp_up'] = aws_up.temp
# aws_comb['temp_far'] = aws_far.temp
# # aws_comb = aws_comb.resample('m').mean()
#
# plt.plot(aws_comb)
# plt.show()
