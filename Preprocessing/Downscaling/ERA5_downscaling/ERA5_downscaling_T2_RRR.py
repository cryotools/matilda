import datetime; starttime = datetime.datetime.now().time(); print(starttime)
from pathlib import Path ; home = str(Path.home())
import sys; sys.path.append('/home/ana/Seafile/SHK/Scripts/centralasiawaterresources/Preprocessing/Downscaling/ERA5_downscaling')
import numpy as np; import pandas as pd; import xarray as xr; import salem
from misc_functions.xarray_functions import insert_var
from misc_functions.inspect_values import mmm_nan, mmm_nan_name, check, check_for_nans_dataframe, calculate_r2_and_rmse
from fundamental_physical_constants import g, M, R, teten_a1, teten_a3, teten_a4, zero_temperature
from misc_functions.calculate_parameters import calculate_ew; import matplotlib.pyplot as plt

working_directory = home + ""
shape_file = ""
era5_static_file = home + ""

era5_file = working_directory + ""
output_csv = working_directory + ""

target_altitude = 0
margin = 0.2
z0 = 0.00212                                # (m) mean between roughness firn 4 mm and fresh snow 0.24 mm
lapse_rate_temperature = -0.006             # K/m  temperature lapse rate

#Time slice
time_start = '2001-01-01T01:00'
time_end = '2020-11-01T01:00'

era5_static = salem.open_xr_dataset(era5_static_file)
#shape_grid = salem.read_shapefile_to_grid(shape_file,grid=salem.grid_from_dataset(era5_static))
era5 = salem.open_xr_dataset(era5_file)
#lon_ll = shape_grid.CenLon.values - margin; lat_ll = shape_grid.CenLat.values - margin; lon_ur = shape_grid.CenLon.values + margin; lat_ur = shape_grid.CenLat.values + margin
#era5_static = era5_static.salem.subset(corners=((lon_ll,lat_ll), (lon_ur,lat_ur)), crs=salem.wgs84) ###ds1 = ds1.salem.roi(shape=shape,all_touched=False) #ds1 = ds.salem.subset(shape=shape, margin=5)

era5 = era5.sel(time=slice(time_start, time_end))

# fig = plt.figure(figsize=(12, 6))
# smap = era5_static.salem.get_map(data=era5_static.z/g, cmap="topo") #, vmin=0, vmax=6000)
# smap.set_shapefile(shape_file)
# smap.visualize()
# plt.show()

## Select closest gridpoint
# lon_distances_gp = np.abs(era5_static.lon.values-shape_grid.CenLon.values[0])
# lat_distances_gp = np.abs(era5_static.lat.values-shape_grid.CenLat.values[0])
# idx_lon = np.where(lon_distances_gp == np.nanmin(lon_distances_gp))
# idx_lat = np.where(lat_distances_gp == np.nanmin(lat_distances_gp))
# latitude = float(era5_static.lat[idx_lat].values)
# longitude = float(era5_static.lon[idx_lon].values)

latitude = 41
longitude = 75.9

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
height_diff = target_altitude - era5_static.z.values/g ; print("Height difference between target_altitude: ", height_diff)
print('First timestamp: ', era5.time[0].values, ' last timestamp: ', era5.time[-1].values)
print("Altitude of gridpoint ", era5_static.z.values/g)

temperature = (era5['t2m'].values) + height_diff * float(lapse_rate_temperature)


##############################%^%%%%%%IMPORT
### TP, strd and ssrd cumulative value over 24 hours, therefore use diff, only for values at midnight for that use orginal value
total_precipitation = np.append(0, (era5.tp.diff(dim='time').values.flatten())) #+ height_diff * lapse_rate_total_precipitation))         ### convert from m to mm
total_precipitation[total_precipitation < 0] = era5.tp.values[total_precipitation < 0]
total_precipitation[total_precipitation < 0] = 0
total_precipitation = total_precipitation * 1000


### delete first value because of problem with accumualted variables
time_local = era5['time'].to_index()
time_local = time_local.tz_localize('Asia/Bishkek')
time_local = time_local[1:]
temperature = temperature[1:]
total_precipitation = total_precipitation[1:]

print('T2'); mmm_nan(temperature); mmm_nan(era5['t2m'].values)
print('TP'); mmm_nan(total_precipitation); mmm_nan(era5['tp'].values)
check(temperature,'T2',316.16,225.16)
check(total_precipitation,'TP',25.0,0.0)

raw_data = {'TIMESTAMP': time_local, 'T2': temperature, 'RRR': total_precipitation}
df = pd.DataFrame(raw_data, columns = ['TIMESTAMP', 'T2',  'RRR'])
#df.to_csv(output_csv,index=False)
