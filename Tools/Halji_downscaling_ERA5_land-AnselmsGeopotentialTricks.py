import datetime; starttime = datetime.datetime.now().time(); print(starttime)
from pathlib import Path ; home = str(Path.home())
import sys; sys.path.append('/home/anz/Seafile/work/software/python_source_code/preprocessing_cosipy/Anselm/')
import numpy as np; import pandas as pd; import xarray as xr; import salem
from misc_functions.xarray_functions import insert_var
from misc_functions.inspect_values import mmm_nan, mmm_nan_name, check, check_for_nans_dataframe, calculate_r2_and_rmse
from fundamental_physical_constants import g, M, R, teten_a1, teten_a3, teten_a4, zero_temperature
from misc_functions.calculate_parameters import calculate_ew; import matplotlib.pyplot as plt

working_directory = home + '/Seafile/work/io/Halji/'
era5_land_static_file = home + '/Seafile/work/io/ERA5_land/ERA5_land_Z_geopotential.nc'
era5_land_file = working_directory + '/ERA5_land/20200302_Halji_ERA5_land_1981_201911.nc'
aws_file = working_directory  + 'AWS/20191104_AWS_Halji.nc'
shape_file = working_directory + '/static/Shapefiles/Halji_RGI6.shp'

target_altitude = 5359
timezone_difference_to_UTC = 6
margin = 0.2
z0 = 0.00212                                # (m) mean between roughness firn 4 mm and fresh snow 0.24 mm
find_point_and_calculate_data = True
store_netCDF = True

era5_land_static = salem.open_xr_dataset(era5_land_static_file)
shape_grid = salem.read_shapefile_to_grid(shape_file,grid=salem.grid_from_dataset(era5_land_static))
era5_land = salem.open_xr_dataset(era5_land_file)
aws = xr.open_dataset(aws_file)         ###2018-04-19T02:00:00 ... 2019-11-03T10:00:00 AWS time
lon_ll = shape_grid.CenLon.values - margin; lat_ll = shape_grid.CenLat.values - margin; lon_ur = shape_grid.CenLon.values + margin; lat_ur = shape_grid.CenLat.values + margin
era5_land_static = era5_land_static.salem.subset(corners=((lon_ll,lat_ll), (lon_ur,lat_ur)), crs=salem.wgs84) ###ds1 = ds1.salem.roi(shape=shape,all_touched=False) #ds1 = ds.salem.subset(shape=shape, margin=5)

fig = plt.figure(figsize=(12, 6))
smap = era5_land_static.salem.get_map(data=era5_land_static.z/g, cmap="topo") #, vmin=0, vmax=6000)
smap.set_shapefile(shape_file)
smap.visualize()
plt.show()

## Select closest gridpoint
lon_distances_gp = np.abs(era5_land_static.lon.values-shape_grid.CenLon.values)
lat_distances_gp = np.abs(era5_land_static.lat.values-shape_grid.CenLat.values)
idx_lon = np.where(lon_distances_gp == np.nanmin(lon_distances_gp))
idx_lat = np.where(lat_distances_gp == np.nanmin(lat_distances_gp))
latitude = float(era5_land_static.lat[idx_lat].values)
longitude = float(era5_land_static.lon[idx_lon].values)

#### Select closest gridpoint in contrast to elevation
#altitude_differences_gp = np.abs(era5_land_static.z/g - target_altitude)
#latitude = float(era5_land_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lat)
#longitude = float(era5_land_static.where(altitude_differences_gp == np.nanmin(altitude_differences_gp), drop=True).lon)

## Select highest gridpoint
# latitude = float(era5_land_static.where(era5_land_static == era5_land_static.max(), drop=True).lat)
# longitude = float(era5_land_static.where(era5_land_static == era5_land_static.max(), drop=True).lon)

# ## Select lowest gridpoint
# latitude = float(era5_land_static.where(era5_land_static == era5_land_static.min(), drop=True).lat)
# longitude = float(era5_land_static.where(era5_land_static == era5_land_static.min(), drop=True).lon)

output_csv = working_directory + 'input/20200305_Halji_ERA5_land_1981_20191130_closest_gp.csv'

## Subset time
#time_start = aws.time[0]; time_end = aws.time[-1]; #time_start = '2010-10-01'; time_end = '2019-09-30'
#era5_land = era5_land.sel(time=slice(time_start,time_end)); print(era5_land.time.values[0]); print(era5_land.time.values[-1])
era5_land = era5_land.sel(lat=latitude, lon=longitude, method='nearest'); era5_land_static = era5_land_static.sel(lat=latitude,lon=longitude, method='nearest')
era5_land.t2m.to_pandas().to_csv(working_directory + 'QMAP/20200210_Halji_ERA5_land_T2_2000_2019_closest_gp.csv')
height_diff = target_altitude  - era5_land_static.z.values/g ; print("Height difference between target_altitude: ", height_diff)
print('First timestamp: ', era5_land.time[0].values, ' last timestamp: ', era5_land.time[-1].values)
print("Altitude of gridpoint ", era5_land_static.z.values/g)

aws_start = aws.time[0]         ### '2018-10-01'
aws_end = min(aws.time[-1], era5_land.time[-1]) ### full periode
#aws_end = '2019-04-19T02:00'    ### only one year
aws = aws.sel(time=slice(aws_start, aws_end))
era5_land_compare = era5_land.sel(time=slice(aws_start, aws_end))

measured_temperatures = aws.T2[~np.isnan(aws.T2)]
simulated_temperatures = era5_land_compare.t2m.values[~np.isnan(aws.T2)]

measured_rh2 = aws.RH2[~np.isnan(aws.RH2)]
es_d2m_compare = teten_a1 * np.exp(teten_a3 * (era5_land_compare['d2m'].values-zero_temperature)/(era5_land_compare['d2m'].values-teten_a4))
es_t2m_compare = teten_a1 * np.exp(teten_a3 * (era5_land_compare['t2m'].values-zero_temperature)/(era5_land_compare['t2m'].values-teten_a4))
relative_humidity_compare = 100 * (es_d2m_compare)/(es_t2m_compare)
simulated_rh2 = relative_humidity_compare[~np.isnan(aws.RH2)]

measured_pressure = aws.PRES[~np.isnan(aws.PRES)]
simulated_pressure = (era5_land_compare['sp'].values/100)[~np.isnan(aws.PRES)]

measured_G = aws.G[~np.isnan(aws.G)]
simulated_G = np.append(0, era5_land_compare.ssrd.diff(dim='time').values/(60*60))[~np.isnan(aws.G)]
simulated_G[simulated_G < 0] = 0

measured_U2 = aws.U2[~np.isnan(aws.U2)]
U10_compare = np.sqrt(era5_land_compare['v10'].values**2 + era5_land_compare['u10'].values**2)
simulated_U2 = (U10_compare * (np.log(2/z0)/np.log(10/z0)))[~np.isnan(aws.U2)]

aws_tp = xr.open_dataset(working_directory + 'AWS/20191104_AWS_Halji_tp.nc')
start_date = max(aws_tp.time[0], era5_land_compare.time[0])
end_date = min(aws_tp.time[-1], era5_land_compare.time[-1])
aws_tp = aws_tp.sel(time=slice(start_date, end_date))
era5_land_compare = era5_land.sel(time=slice(start_date, end_date))

measured_TP = aws_tp.RRR[~np.isnan(aws_tp.RRR)]
simulated_TP = np.append(0, (era5_land_compare.tp.diff(dim='time').values.flatten() * 1000))[~np.isnan(aws_tp.RRR)]
simulated_TP[simulated_TP<0] = 0

print('before appling downscliang: ', datetime.datetime.now().time())
if find_point_and_calculate_data is True:
    ### R solution
    data = {'measured_temperatures': measured_temperatures, 'simulated_temperatures': simulated_temperatures}
    df_for_R = pd.DataFrame(data, columns=['measured_temperatures', 'simulated_temperatures'])
    df_for_R.to_csv(working_directory + 'QMAP/temperatures_for_qmap.csv', index=False)
    temperature = pd.read_csv(working_directory + 'QMAP/mapped_T2.csv').values[:,0]
    temperature[temperature < 200] = np.min(temperature[temperature > 200])
    lapse_rate_temperature = (np.mean(measured_temperatures) - np.mean(simulated_temperatures)) / height_diff; print('downscaling PRES: ', lapse_rate_temperature)
    temperature_lapse = (era5_land['t2m'].values) + height_diff * float(lapse_rate_temperature)
    #lapse_rate_temperature = -0.006             # K/m  temperature lapse rate
    #print('lapse rate temperature plus height diff: ', lapse_rate_temperature.values + height_diff)
    #temperature = (era5_land['t2m'].values) + height_diff * float(lapse_rate_temperature)
    #temperature[temperature < 200] = np.nan

    ### Python solution
    # quantile_measured = stats.norm.cdf(measured_temperatures, loc=np.mean(measured_temperatures), scale=np.std(measured_temperatures))
    # temperature = stats.norm.ppf(quantile_measured, loc=np.mean(simulated_temperatures), scale=np.std(simulated_temperatures))
    ### other possibilty: z_transform_t2 = stats.zscore(measured_temperatures); quantile_z_transformed = stats.norm.cdf(z_transform_t2, loc=0, scale=1)

    lapse_rate_pres = (np.mean(measured_pressure) - np.mean(simulated_pressure)) / height_diff; print('downscaling PRES: ', lapse_rate_pres)
    air_pressure = ((era5_land['sp'].values)/100) + height_diff * float(lapse_rate_pres)
    # air_pressure = ((era5_land['sp'].values)/100) * (1-abs(lapse_rate_temperature)*height_diff/era5_land['t2m'].values) ** ((g*M)/(R*abs(lapse_rate_temperature)))

    scaling_wind = (np.mean(measured_U2) / np.mean(simulated_U2)); print("Scaling wind: ", scaling_wind)
    U10 = np.sqrt(era5_land['v10'].values ** 2 + era5_land['u10'].values ** 2)
    U2 = U10 * (np.log(2 / z0) / np.log(10 / z0)) * float(scaling_wind)

    lapse_rate_rh2 = (np.mean(measured_rh2) - np.mean(simulated_rh2)) / height_diff; print('downscaling RH2: ', lapse_rate_rh2)
    es_d2m = teten_a1 * np.exp(teten_a3 * (era5_land['d2m'].values-zero_temperature)/(era5_land['d2m'].values-teten_a4))
    es_t2m = teten_a1 * np.exp(teten_a3 * (era5_land['t2m'].values-zero_temperature)/(era5_land['t2m'].values-teten_a4))
    relative_humidity = 100 * (es_d2m)/(es_t2m) + height_diff * float(lapse_rate_rh2)

    total_precipitation_scaling = 2.1010939
    total_precipitation = np.append(0, (era5_land.tp.diff(dim='time').values.flatten() * 1000 * total_precipitation_scaling)) #+ height_diff * lapse_rate_total_precipitation))         ### convert from m to mm
    shortwave_in = np.append(0, era5_land.ssrd.diff(dim='time').values/(60*60))
    longwave_in = np.append(0, era5_land.strd.diff(dim='time').values/(60*60))

    relative_humidity[relative_humidity>100] = 100
    total_precipitation[total_precipitation<0] = 0
    shortwave_in[shortwave_in<0] = 0
    longwave_in[longwave_in<0] = 0

    print('T2'); mmm_nan(temperature); mmm_nan(era5_land['t2m'].values)
    print('PRES'); mmm_nan(air_pressure); mmm_nan(era5_land['sp'].values)
    print('U2'); mmm_nan(U2); mmm_nan(U10)
    print('RH2'); mmm_nan(relative_humidity)
    print('TP'); mmm_nan(total_precipitation); mmm_nan(era5_land['tp'].values)
    print('G'); mmm_nan(shortwave_in)
    print('LWin'); mmm_nan(longwave_in)
    check(temperature,'T2',316.16,225.16)
    check(air_pressure,'PRES',700.0,500.0)
    check(U2,'U2',50.0,0.0)
    check(relative_humidity,'RH2',100.0,0.0)
    check(total_precipitation,'TP',25.0,0.0)
    check(shortwave_in,'G',1600.0,0.0)
    check(longwave_in,'LWin',400.0,0.0)
    time_local = era5_land['time'].to_index() + pd.Timedelta(hours=timezone_difference_to_UTC)
    raw_data = {'TIMESTAMP': time_local, 'T2': temperature, 'PRES': air_pressure, 'U2': U2,
                'RH2':relative_humidity, 'RRR': total_precipitation, 'G': shortwave_in, 'LWin': longwave_in
                }
    df = pd.DataFrame(raw_data, columns = ['TIMESTAMP', 'T2', 'PRES', 'U2', 'RH2', 'RRR', 'G', 'LWin'])
    check_for_nans_dataframe(df)
    df.to_csv(output_csv,index=False)
    print("CSV file has been stored to disc")
else:
    df = pd.read_csv(output_csv, delimiter=',', index_col=['TIMESTAMP'], parse_dates=['TIMESTAMP'])

print('after appling downscliang: ', datetime.datetime.now().time())

ds_era = xr.Dataset()
ds_era.coords['time'] = era5_land.time #+ pd.Timedelta(hours=timezone_difference_to_UTC)
insert_var(ds_era, df['T2'], 'T2', 'K', 'Air temperature 2m')
insert_var(ds_era, df['PRES'], 'PRES', 'hPa', 'Air pressure')
insert_var(ds_era, df['RRR'], 'RRR', 'mm', 'Total precipitation')
insert_var(ds_era, df['G'], 'G', 'W m\u207b\xb2', 'Incoming shortwave radiation')
insert_var(ds_era, df['U2'], 'U2', 'm s\u207b\xb9', 'Wind speed 2m')
insert_var(ds_era, df['RH2'], 'RH2', '%', 'Relative humidity 2m')
insert_var(ds_era, df['LWin'], 'LWin', 'W m\u207b\xb2', 'Incoming longwave radiation')

if store_netCDF is True:
    output_netcdf = working_directory + 'ERA5_land/' + (output_csv.split('/input/')[1]).split('csv')[0] + 'nc'
    ds_era.to_netcdf(output_netcdf)

from misc_functions.plot_functions import plot_line_diff, scatterplot_linear, plot_cdf, plot_cdf_compare_3var, plot_cdf_compare_4var, plot_pdf_3var
plt_dir = working_directory + 'plots/comparison_AWS_datasets/0304/' + (output_csv.split('/input/')[1]).split('.csv')[0] + '/'; print(plt_dir)
dataset_name = 'ERA5_land'

es_aws = calculate_ew(aws.T2.values)            ### FROM https://earthscience.stackexchange.com/questions/2360/how-do-i-convert-specific-humidity-to-relative-humidity
saturation_mixing_ratio_vapour_aws = 0.622 * (es_aws/((aws.PRES.values)*100))
specific_humidity_aws = (aws.RH2.values/100) * saturation_mixing_ratio_vapour_aws
insert_var(aws, specific_humidity_aws * 1000, 'SH', 'g/kg', 'Specific humidity')

es_era = calculate_ew(ds_era.T2.values)
saturation_mixing_ratio_vapour_era = 0.622 * (es_era/(ds_era.PRES.values*100))
specific_humidity_era = (ds_era.RH2.values/100) * saturation_mixing_ratio_vapour_era
insert_var(ds_era, specific_humidity_era * 1000, 'SH', 'g/kg', 'Specific humidity')

import os; os.mkdir(plt_dir)
start_date = None; end_date = None; integration = 'h'
start_date = max(aws.time[0], ds_era.time[0])
end_date = min(aws.time[-1], ds_era.time[-1])
if integration == None:
   integration = 'h'
if start_date != None:
    print('Slice dataset')
    aws = aws.sel(time=slice(start_date, end_date))
    ds_era = ds_era.sel(time=slice(start_date, end_date))

def plot_compare_scatter_era_aws(field, line_best_fit = False):
    scatterplot_linear(ds_era[field].resample(time=integration).mean(), aws[field].resample(time=integration).mean(),
                    dataset_name, 'AWS', ds_era[field].long_name, integration, plt_dir, str(integration), ds_era[field].units, line_best_fit = line_best_fit)
    calculate_r2_and_rmse(ds_era[field].resample(time=integration).mean(), aws[field].resample(time=integration).mean(), aws[field].name)

plot_compare_scatter_era_aws('U2', line_best_fit = True)
plot_compare_scatter_era_aws('RH2', line_best_fit = True)
plot_compare_scatter_era_aws('T2', line_best_fit = True)
plot_compare_scatter_era_aws('PRES', line_best_fit = True)
plot_compare_scatter_era_aws('G', line_best_fit = True)
plot_compare_scatter_era_aws('SH', line_best_fit = True)

def compare_mean(var):
    print(var, 'dataset : ', np.nanmean(ds_era[var]), ' AWS :', np.nanmean(aws[var]))
compare_mean('T2'); compare_mean('RH2'); compare_mean('PRES'); compare_mean('G'); compare_mean('U2')

#plot_cdf_compare_5var(measured_temperatures, simulated_temperatures, ds_era.T2[~np.isnan(aws.T2)], df['T2'], temperature_lapse, aws.T2.long_name, aws.T2.long_name, plt_dir, name=False, save=True)
def plot_cdf_curves(measured, simulated, field, temperature='K'):
    plot_cdf_compare_3var(measured, simulated, ds_era[field][~np.isnan(aws[field])], ds_era[field].units, ds_era[field].long_name, plt_dir, name=False, save=True)
    plot_cdf_compare_4var(measured, simulated, ds_era[field][~np.isnan(aws[field])], df[field], aws[field].units, aws[field].long_name, plt_dir, name=False, save=True)
    #plot_pdf_3var(measured, simulated, ds_era[field][~np.isnan(aws[field])], ds_era[field].long_name, ds_era[field].long_name, plt_dir, name=False, save=True)
plot_cdf_curves(measured_temperatures, simulated_temperatures, 'T2')
plot_cdf_curves(measured_G, simulated_G, 'G')
plot_cdf_curves(measured_rh2, simulated_rh2, 'RH2')
plot_cdf_curves(measured_pressure, simulated_pressure, 'PRES')
plot_cdf_curves(measured_U2, simulated_U2, 'U2')
plot_cdf_curves(measured_temperatures, simulated_temperatures, 'T2')

def plot_compare_era_aws(field,temperature='K'):
    plot_line_diff(ds_era[field].resample(time=integration).mean(),aws[field].resample(time=integration).mean(),
                   dataset_name, 'AWS', ds_era[field].long_name, str(integration), ds_era[field].units, plt_dir, integration,temperature=temperature)
plot_compare_era_aws('T2', temperature = 'C')
plot_compare_era_aws('RH2')
plot_compare_era_aws('PRES')
plot_compare_era_aws('G')
plot_compare_era_aws('U2')
plot_compare_era_aws('SH')

print("AWS"); mmm_nan_name(aws.T2); print("ERA5"); mmm_nan_name(ds_era.T2)
print("AWS"); mmm_nan_name(aws.RH2); print("ERA5"); mmm_nan_name(ds_era.RH2)
print("AWS"); mmm_nan_name(aws.PRES); print("ERA5"); mmm_nan_name(ds_era.PRES)
print("AWS"); mmm_nan_name(aws.G); print("ERA5"); mmm_nan_name(ds_era.G)
print("AWS"); mmm_nan_name(aws.U2); print("ERA5"); mmm_nan_name(ds_era.U2)

aws_tp = xr.open_dataset(working_directory + 'AWS/20191104_AWS_Halji_tp.nc')
aws_tp_start = max(aws_tp.time[0], ds_era.time[0])
aws_tp_end = min(aws_tp.time[-1], ds_era.time[-1])
aws_tp = aws_tp.sel(time=slice(aws_tp_start, aws_tp_end))
ds_era = ds_era.sel(time=slice(aws_tp_start, aws_tp_end))

field = 'RRR'
#plot_pdf_3var(measured_TP, simulated_TP, ds_era[field][~np.isnan(aws_tp[field])], ds_era[field].long_name, ds_era[field].long_name, plt_dir)
#plot_cdf_compare_3var(measured_TP, simulated_TP, ds_era[field][~np.isnan(aws_tp[field])], ds_era[field].long_name, ds_era[field].long_name, plt_dir)
plot_cdf_compare_4var(measured_TP, simulated_TP, ds_era[field][~np.isnan(aws_tp[field])], df[field], aws_tp[field].long_name, aws_tp[field].long_name, plt_dir)
plot_line_diff(ds_era[field].resample(time=integration).mean(),aws_tp[field].resample(time=integration).mean(), dataset_name, 'AWS', ds_era[field].long_name, str(integration), ds_era[field].units, plt_dir, integration)
scatterplot_linear(ds_era[field].resample(time=integration).mean(), aws_tp[field].resample(time=integration).mean(), dataset_name, 'AWS', ds_era[field].long_name, integration, plt_dir, str(integration), ds_era[field].units, line_best_fit = True)
calculate_r2_and_rmse(ds_era[field].resample(time=integration).mean(), aws_tp[field].resample(time=integration).mean(), aws_tp[field].name)

var1_cum = np.cumsum(ds_era[field])
var2_cum = np.cumsum(aws_tp[field])
plot_line_diff(var1_cum, var2_cum, dataset_name, 'AWS', ds_era[field].long_name, str(integration), ds_era[field].units, plt_dir, 'cumulative')

print('starttime: ', starttime)
print('time start: ', time_local[0], ' time end: ', time_local[-1])
print("All plots done ", datetime.datetime.now().time())
