import datetime; starttime = datetime.datetime.now().time(); print(starttime)
from pathlib import Path ; home = str(Path.home())
import sys; sys.path.append('/home/ana/Seafile/SHK/Scripts/centralasiawaterresources/Preprocessing/ERA5_downscaling')
import numpy as np; import pandas as pd; import xarray as xr; import salem
from misc_functions.xarray_functions import insert_var
from misc_functions.inspect_values import mmm_nan, mmm_nan_name, check, check_for_nans_dataframe, calculate_r2_and_rmse
from fundamental_physical_constants import g, M, R, teten_a1, teten_a3, teten_a4, zero_temperature
from misc_functions.calculate_parameters import calculate_ew; import matplotlib.pyplot as plt

working_directory = home + ""
shape_file = working_directory + ""
era5_static_file = home + ""

era5_file = working_directory + ""
output_csv = working_directory + ""

target_altitude = 0
timezone_difference_to_UTC = 6
margin = 0.2
z0 = 0.00212                                # (m) mean between roughness firn 4 mm and fresh snow 0.24 mm
lapse_rate_temperature = -0.006             # K/m  temperature lapse rate

era5_static = salem.open_xr_dataset(era5_static_file)
shape_grid = salem.read_shapefile_to_grid(shape_file,grid=salem.grid_from_dataset(era5_static))
era5 = salem.open_xr_dataset(era5_file)
lon_ll = shape_grid.CenLon.values - margin; lat_ll = shape_grid.CenLat.values - margin; lon_ur = shape_grid.CenLon.values + margin; lat_ur = shape_grid.CenLat.values + margin
era5_static = era5_static.salem.subset(corners=((lon_ll,lat_ll), (lon_ur,lat_ur)), crs=salem.wgs84) ###ds1 = ds1.salem.roi(shape=shape,all_touched=False) #ds1 = ds.salem.subset(shape=shape, margin=5)

# fig = plt.figure(figsize=(12, 6))
# smap = era5_static.salem.get_map(data=era5_static.z/g, cmap="topo") #, vmin=0, vmax=6000)
# smap.set_shapefile(shape_file)
# smap.visualize()
# plt.show()

## Select closest gridpoint
lon_distances_gp = np.abs(era5_static.lon.values-shape_grid.CenLon.values[0])
lat_distances_gp = np.abs(era5_static.lat.values-shape_grid.CenLat.values[0])
idx_lon = np.where(lon_distances_gp == np.nanmin(lon_distances_gp))
idx_lat = np.where(lat_distances_gp == np.nanmin(lat_distances_gp))
latitude = float(era5_static.lat[idx_lat].values)
longitude = float(era5_static.lon[idx_lon].values)

# ### Select closest gridpoint in contrast to elevation
# altitude_differences_gp = np.abs(era5_static.z/g - target_altitude)
# latitude = float(era5_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lat)
# longitude = float(era5_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lon)
#
# # Select highest gridpoint
# latitude = float(era5_static.where(era5_static == era5_static.max(), drop=True).lat)
# longitude = float(era5_static.where(era5_static == era5_static.max(), drop=True).lon)
#
# ## Select lowest gridpoint
# latitude = float(era5_static.where(era5_static == era5_static.min(), drop=True).lat)
# longitude = float(era5_static.where(era5_static == era5_static.min(), drop=True).lon)

era5 = era5.sel(lat=latitude, lon=longitude, method='nearest'); era5_static = era5_static.sel(lat=latitude,lon=longitude, method='nearest')
height_diff = target_altitude  - era5_static.z.values/g ; print("Height difference between target_altitude: ", height_diff)
print('First timestamp: ', era5.time[0].values, ' last timestamp: ', era5.time[-1].values)
print("Altitude of gridpoint ", era5_static.z.values/g)

temperature = (era5['t2m'].values) + height_diff * float(lapse_rate_temperature)
air_pressure = ((era5['sp'].values)/100) * (1-abs(lapse_rate_temperature)*height_diff/era5['t2m'].values) ** ((g*M)/(R*abs(lapse_rate_temperature)))
scaling_wind = 2.0
U10 = np.sqrt(era5['v10'].values ** 2 + era5['u10'].values ** 2)
U2 = U10 * (np.log(2 / z0) / np.log(10 / z0)) * float(scaling_wind)

es_d2m = teten_a1 * np.exp(teten_a3 * (era5['d2m'].values-zero_temperature)/(era5['d2m'].values-teten_a4))
es_t2m = teten_a1 * np.exp(teten_a3 * (era5['t2m'].values-zero_temperature)/(era5['t2m'].values-teten_a4))
relative_humidity = 100 * (es_d2m)/(es_t2m)

### TP, strd and ssrd cumulative value over 24 hours, therefore use diff, only for values at midnight for that use orginal value
total_precipitation = np.append(0, (era5.tp.diff(dim='time').values.flatten())) #+ height_diff * lapse_rate_total_precipitation))         ### convert from m to mm
total_precipitation[total_precipitation < 0] = era5.tp.values[total_precipitation < 0]
total_precipitation[total_precipitation < 0] = 0
total_precipitation = total_precipitation * 1000

longwave_in = np.append(0, era5.strd.diff(dim='time').values / (60 * 60))
longwave_in[longwave_in < 0] = (era5.strd.values / (60 * 60))[longwave_in < 0]
#longwave_in[0] = era5.strd.values[0] / (60*60)

shortwave_in = np.append(0, era5.ssrd.diff(dim='time').values/(60*60))
shortwave_in[longwave_in<0] = (era5.ssrd.values / (60 * 60))[longwave_in<0]
shortwave_in[shortwave_in<0] = 0

relative_humidity[relative_humidity>100] = 100.
relative_humidity[relative_humidity < 0] = 0.0

### delete first value because of problem with accumualted variables
time_local = era5['time'].to_index() + pd.Timedelta(hours=timezone_difference_to_UTC)
time_local = time_local[1:]
temperature = temperature[1:]
air_pressure = air_pressure[1:]
U2 = U2[1:]
relative_humidity = relative_humidity[1:]
total_precipitation = total_precipitation[1:]
shortwave_in = shortwave_in[1:]
longwave_in = longwave_in[1:]

print('T2'); mmm_nan(temperature); mmm_nan(era5['t2m'].values)
print('PRES'); mmm_nan(air_pressure); mmm_nan(era5['sp'].values)
print('U2'); mmm_nan(U2); mmm_nan(U10)
print('RH2'); mmm_nan(relative_humidity)
print('TP'); mmm_nan(total_precipitation); mmm_nan(era5['tp'].values)
print('G'); mmm_nan(shortwave_in)
print('LWin'); mmm_nan(longwave_in)
check(temperature,'T2',316.16,225.16)
check(air_pressure,'PRES',700.0,500.0)
check(U2,'U2',10.0,0.0)
check(relative_humidity,'RH2',100.0,0.0)
check(total_precipitation,'TP',25.0,0.0)
check(shortwave_in,'G',1600.0,0.0)
check(longwave_in,'LWin',400.0,0.0)

raw_data = {'TIMESTAMP': time_local, 'T2': temperature, 'PRES': air_pressure, 'U2': U2,
            'RH2':relative_humidity, 'RRR': total_precipitation, 'G': shortwave_in, 'LWin': longwave_in
            }
df = pd.DataFrame(raw_data, columns = ['TIMESTAMP', 'T2', 'PRES', 'U2', 'RH2', 'RRR', 'G', 'LWin'])
check_for_nans_dataframe(df)
df.to_csv(output_csv,index=False)
print("CSV file has been stored to disc")
