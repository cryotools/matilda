import datetime; starttime = datetime.datetime.now().time(); print(starttime)
from pathlib import Path ; home = str(Path.home())
import rpy2.robjects as robjects
import sys
#import os
#os.chdir('/home/ana/Seafile/Azamat_AvH/data/scripts')
import numpy as np; import pandas as pd; import xarray as xr; import salem
from misc_functions.xarray_functions import insert_var
from misc_functions.inspect_values import mmm_nan, mmm_nan_name, check, check_for_nans_dataframe, calculate_r2_and_rmse
from fundamental_physical_constants import g, M, R, teten_a1, teten_a3, teten_a4, zero_temperature
from misc_functions.calculate_parameters import calculate_ew; import matplotlib.pyplot as plt

working_directory = home + '/Seafile/Azamat_AvH/data/test.run_karabatkak'
shape_file = working_directory + '/static/shp/kb_outline_rgi_4326.shp'
era5_land_static_file = working_directory + '/static/ERA5_global_z.nc'

era5_land_file = working_directory + '/era5/20200722_Karabatkak_ERA5L_1982_2019.nc'
aws_file = working_directory + "/input/obs_kyzylsuu_aws_2008_2011_19.csv"
output_csv = working_directory + '/input/obs_202102181_kyzylsuu_awsq_2008_2011.csv'

start_date = '2008-01-01 00:00:00'
end_date = '2018-12-31 23:00:00'
target_altitude = 2550
timezone_difference_to_UTC = 6
margin = 0.2
z0 = 0.00212                                # (m) mean between roughness firn 4 mm and fresh snow 0.24 mm
lapse_rate_temperature = -0.006             # K/m  temperature lapse rate

era5_land_static = salem.open_xr_dataset("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/global/ERA5_global_z.nc")
shape_grid = salem.read_shapefile_to_grid(shape_file,grid=salem.grid_from_dataset(era5_land_static))
era5_land = xr.open_dataset(era5_land_file)
#era5_land = salem.open_xr_dataset(era5_land_file)
lon_ll = shape_grid.CenLon.values - margin; lat_ll = shape_grid.CenLat.values - margin; lon_ur = shape_grid.CenLon.values + margin; lat_ur = shape_grid.CenLat.values + margin
era5_land_static = era5_land_static.salem.subset(corners=((lon_ll,lat_ll), (lon_ur,lat_ur)), crs=salem.wgs84) ###ds1 = ds1.salem.roi(shape=shape,all_touched=False) #ds1 = ds.salem.subset(shape=shape, margin=5)
aws = pd.read_csv(aws_file)
#aws["TIMESTAMP"] = pd.to_datetime(aws[['year', 'month', 'day']])
aws["TIMESTAMP"] = pd.to_datetime(aws["TIMESTAMP"])
aws.set_index('TIMESTAMP', inplace=True)
aws = aws.loc[start_date: end_date]
aws["T2"] = pd.to_numeric(aws.T2, errors='coerce')
aws["RRR"] = pd.to_numeric(aws.RRR, errors='coerce')


# fig = plt.figure(figsize=(12, 6))
# smap = era5_land_static.salem.get_map(data=era5_land_static.z/g, cmap="topo") #, vmin=0, vmax=6000)
# smap.set_shapefile(shape_file)
# smap.visualize()
# plt.show()

## Select closest gridpoint
lon_distances_gp = np.abs(era5_land_static.lon.values-shape_grid.CenLon.values)
lat_distances_gp = np.abs(era5_land_static.lat.values-shape_grid.CenLat.values)
idx_lon = np.where(lon_distances_gp == np.nanmin(lon_distances_gp))
idx_lat = np.where(lat_distances_gp == np.nanmin(lat_distances_gp))
latitude = float(era5_land_static.lat[idx_lat].values)
longitude = float(era5_land_static.lon[idx_lon].values)

# ### Select closest gridpoint in contrast to elevation
# altitude_differences_gp = np.abs(era5_land_static.z/g - target_altitude)
# latitude = float(era5_land_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lat)
# longitude = float(era5_land_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lon)
#
# # Select highest gridpoint
# latitude = float(era5_land_static.where(era5_land_static == era5_land_static.max(), drop=True).lat)
# longitude = float(era5_land_static.where(era5_land_static == era5_land_static.max(), drop=True).lon)
#
# ## Select lowest gridpoint
# latitude = float(era5_land_static.where(era5_land_static == era5_land_static.min(), drop=True).lat)
# longitude = float(era5_land_static.where(era5_land_static == era5_land_static.min(), drop=True).lon)

era5_land = era5_land.sel(lat=latitude, lon=longitude, method='nearest'); era5_land_static = era5_land_static.sel(lat=longitude, lon=latitude, method='nearest')
height_diff = target_altitude - era5_land_static.z.values/g ; print("Height difference between target_altitude: ", height_diff)
era5_land = era5_land.sel(time=slice(start_date, end_date))
print('First timestamp: ', era5_land.time[0].values, ' last timestamp: ', era5_land.time[-1].values)
print("Altitude of gridpoint ", era5_land_static.z.values/g)

air_pressure = ((era5_land['sp'].values)/100) * (1-abs(lapse_rate_temperature)*height_diff/era5_land['t2m'].values) ** ((g*M)/(R*abs(lapse_rate_temperature)))
scaling_wind = 2.0
U10 = np.sqrt(era5_land['v10'].values ** 2 + era5_land['u10'].values ** 2)
U2 = U10 * (np.log(2 / z0) / np.log(10 / z0)) * float(scaling_wind)

es_d2m = teten_a1 * np.exp(teten_a3 * (era5_land['d2m'].values-zero_temperature)/(era5_land['d2m'].values-teten_a4))
es_t2m = teten_a1 * np.exp(teten_a3 * (era5_land['t2m'].values-zero_temperature)/(era5_land['t2m'].values-teten_a4))
relative_humidity = 100 * (es_d2m)/(es_t2m)

### TP, strd and ssrd cumulative value over 24 hours, therefore use diff, only for values at midnight for that use orginal value
total_precipitation = np.append(0, (era5_land.tp.diff(dim='time').values.flatten())) #+ height_diff * lapse_rate_total_precipitation))         ### convert from m to mm
total_precipitation[total_precipitation < 0] = era5_land.tp.values[total_precipitation < 0]
total_precipitation[total_precipitation < 0] = 0
total_precipitation = total_precipitation * 1000

# load obs and ERA5 into R to apply quantile mapping: all variables adjusted to daily values to match obs
era5_for_R = pd.DataFrame({"TIMESTAMP":era5_land.time, "temp_era":era5_land["t2m"], "precip_era":total_precipitation, "humidity_era":relative_humidity})
era5_for_R.set_index("TIMESTAMP", inplace=True)
era5_for_R.index = pd.to_datetime(era5_for_R.index)
era5_for_R = era5_for_R.resample("D").agg({"temp_era":"mean", "precip_era":"sum", "humidity_era":"mean"})
if "RH2" in aws.columns:
    df_for_R = pd.DataFrame(aws, index=aws.index, columns=["T2", "RRR", "RH2"])
    df_for_R.columns = ["temp_a", "precip", "humidity"]
else:
    df_for_R = pd.DataFrame(aws, index=aws.index, columns=["T2", "RRR"])
    df_for_R.columns =["temp_a", "precip"]
qmap_start = df_for_R.index[0]; qmap_end = df_for_R.index[-1]
df_for_R["temp_a"] = np.where(df_for_R["temp_a"]<100, df_for_R["temp_a"]+273.15, df_for_R["temp_a"])
era5_for_R_subset = era5_for_R[qmap_start:qmap_end]
df_for_R = df_for_R.merge(era5_for_R_subset, left_index=True, right_index=True)
df_for_R.to_csv(working_directory+"/R_QMAP/dataframe_qmap_subset.csv")
era5_for_R.to_csv(working_directory+"/R_QMAP/dataframe_qmap_era5_total.csv")
r_source = robjects.r['source']
r_source(working_directory+"/R_QMAP/quantile_mapping.R")
fitted = pd.read_csv(working_directory+"/R_QMAP/dataframe_fitted.csv")
fitted["TIMESTAMP"] = pd.to_datetime(fitted["TIMESTAMP"])

temperature = fitted["T2"].to_numpy()
total_precipitation = fitted["RRR"].to_numpy()
if "RH2" in fitted.columns:
    relative_humidity = fitted["RH2"].to_numpy()
#temperature = (era5_land['t2m'].values) + height_diff * float(lapse_rate_temperature)

##############################%^%%%%%%IMPORT
longwave_in = np.append(0, era5_land.strd.diff(dim='time').values / (60 * 60))
longwave_in[longwave_in < 0] = (era5_land.strd.values / (60 * 60))[longwave_in < 0]
#longwave_in[0] = era5_land.strd.values[0] / (60*60)

shortwave_in = np.append(0, era5_land.ssrd.diff(dim='time').values/(60*60))
shortwave_in[longwave_in<0] = (era5_land.ssrd.values / (60 * 60))[longwave_in<0]
shortwave_in[shortwave_in<0] = 0

relative_humidity[relative_humidity>100] = 100.
relative_humidity[relative_humidity < 0] = 0.0

### delete first value because of problem with accumualted variables
time_local = era5_land['time'].to_index() + pd.Timedelta(hours=timezone_difference_to_UTC)
time_local = time_local[1:]
#temperature = temperature[1:]
air_pressure = air_pressure[1:] 
U2 = U2[1:]
if "RH2" not in fitted.columns:
    relative_humidity = relative_humidity[1:]
#total_precipitation = total_precipitation[1:]
shortwave_in = shortwave_in[1:] 
longwave_in = longwave_in[1:]

print('T2'); mmm_nan(temperature); mmm_nan(era5_land['t2m'].values)
print('PRES'); mmm_nan(air_pressure); mmm_nan(era5_land['sp'].values)
print('U2'); mmm_nan(U2); mmm_nan(U10)
print('RH2'); mmm_nan(relative_humidity)
print('TP'); mmm_nan(total_precipitation); mmm_nan(era5_land['tp'].values)
print('G'); mmm_nan(shortwave_in)
print('LWin'); mmm_nan(longwave_in)
check(temperature, "T2", 316.16,225.16)
check(air_pressure,'PRES',700.0,500.0)
check(U2,'U2',10.0,0.0)
check(relative_humidity,'RH2',100.0,0.0)
check(total_precipitation,'RRR',25.0,0.0) # warning because it's now daily instead of hourly
check(shortwave_in,'G',1600.0,0.0)
check(longwave_in,'LWin',400.0,0.0)

if "RH2" in fitted.columns:
    raw_data = {'TIMESTAMP': time_local, 'PRES': air_pressure, 'U2': U2,
            'G': shortwave_in, 'LWin': longwave_in}
    df = pd.DataFrame(raw_data, columns = ['TIMESTAMP', 'PRES', 'U2', 'G', 'LWin'])
    df.set_index("TIMESTAMP", inplace=True)
    df = df[start_date: end_date]
    df.index = pd.to_datetime(df.index)
    df = df.resample("D").agg({"PRES":"mean", "U2":"mean", "G":"mean", "LWin":"mean"})
    df.reset_index(inplace=True)
else:
    raw_data = {'TIMESTAMP': time_local, 'RH2':relative_humidity, 'PRES': air_pressure, 'U2': U2,
            'G': shortwave_in, 'LWin': longwave_in}
    df = pd.DataFrame(raw_data, columns = ['TIMESTAMP', 'RH2', 'PRES', 'U2', 'G', 'LWin'])
    df.set_index("TIMESTAMP", inplace=True)
    df.index = pd.to_datetime(df.index)
    df = df[start_date: end_date]
    df = df.resample("D").agg({"RH2": "mean", "PRES":"mean", "U2":"mean", "G":"mean", "LWin":"mean"})
    df.reset_index(inplace=True)
df = pd.merge(df, fitted)
df = df.loc[:,~df.columns.str.startswith('Unnamed')]
check_for_nans_dataframe(df)
df.to_csv(output_csv,index=False)
print("CSV file has been stored to disc")
