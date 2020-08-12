##
from pathlib import Path; home = str(Path.home())      ### Zieht sich home vom system
import sys; sys.path.append(home + '/Seafile/Phillip_Anselm/scripts/')
import pandas as pd; import xarray as xr; import salem; import numpy as np
from misc_functions.inspect_values import *	# * importiert alle Funktionen in dem Anselms selbstgebauten Funktionsskript. Funktionsnname würde nur die importieren
from fundamental_physical_constants import *
import matplotlib.pyplot as plt

era5_static_file = home + '/Seafile/Ana-Lena_Phillip/data/input_output/ERA5/global/ERA5_global_z.nc'

# Define working directory, input ERA5 file and output csv file
working_directory = '/Seafile/Ana-Lena_Phillip/data/'
shape = home + working_directory + 'input_output/static/Shapefiles/rgi_glacierno1.shp'
era5_file = home + working_directory + 'input_output/ERA5/No1_Urumqi_ERA5_2000_201907.nc'
#Time slice:
time_start = '2011-01-01T00:00'
time_end = '2018-12-31T23:00'
output = home + working_directory + 'input_output/input/202000810_Umrumqi_ERA5_' + time_start.split('-')[0] + '_' + time_end.split('-')[0]   # Heisst input, weil es zwar hier der Output ist aber der COSIPY-Input.
output_cosipy = output + '_cosipy.csv'
output_pypdd = output + '_pypdd.csv'
scaling_wind = 2                    # ERA5-Wind ist tendenziell zu schwach. Scaling mit 2 passt am Urumqi
latitude = 43.00; longitude = 86.75                         ### Urumqi
target_altitude = 4025                                      # m a.s.l Urumqi
timezone_difference_to_UTC = 6


density_fresh_snow = 250.
z0 = 0.00212                                # set roughness length for wind speed 2 m calculation, lapse rates for T2 and rH2, and constants for Magnus equation, (m) mean between roughness firn 4 mm and fresh snow 0.24 mm
lapse_rate_temperature = -0.006             # K/m  temperature lapse rate
lapse_rate_relative_humidity = 0.001        # %/m  relative humidity lapse rate
lapse_rate_snowfall = 0.00
lapse_rate_total_precipitation = 0.00
teten_a1 = 611.21                           # Pa; for Water, ice: 611.21 equations 7.4 and 7.5 from Part IV, Physical processes section
                                            # (Chapter 7, section 7.2.1b) in https://www.ecmwf.int/en/elibrary/16648-part-iv-physical-processes
teten_a3 = 17.502                           # ice: 22.587
teten_a4 = 32.19                            # K; ice: 0.7

ds = salem.open_xr_dataset(era5_static_file)
shape_grid = salem.read_shapefile_to_grid(shape, grid=salem.grid_from_dataset(ds))
margin = 0.2        # abstand vom center des shapefiles
lon_ll = shape_grid.CenLon.values - margin      # ll = lower left
lat_ll = shape_grid.CenLat.values - margin
lon_ur = shape_grid.CenLon.values + margin      # ur = upper right
lat_ur = shape_grid.CenLat.values + margin
print(lon_ll,lat_ll,lon_ur,lat_ur)
ds1 = ds.salem.subset(corners=((lon_ll,lat_ll), (lon_ur,lat_ur)), crs=salem.wgs84)      # ds beinhaltet die geopotentialhöhe aller gridzellen. hier wird für auswahl innerhalb der margin ein subset genommen.
#ds1 = ds1.salem.roi(shape=shape,all_touched=False) #ds1 = ds.salem.subset(shape=shape, margin=5)

## Lage des Shapefiles in den Gitterzellen:
# fig = plt.figure(figsize=(12, 6))
# smap = ds1.salem.get_map(data=ds1.z, cmap="topo") #, vmin=0, vmax=6000)
# smap.set_shapefile(shape) # set shape file
# smap.visualize()
# plt.show()
##

lon_distances_gp = np.abs(ds1.lon.values-shape_grid.CenLon.values[0]) # Werte der Longitude aus dem DS1-Paket MINUS der Longitude des Shape-Centers von einem von beiden Branches (Element 1)
lat_distances_gp = np.abs(ds1.lat.values-shape_grid.CenLat.values[0])
idx_lon = np.where(lon_distances_gp == np.nanmin(lon_distances_gp))
idx_lat = np.where(lat_distances_gp == np.nanmin(lat_distances_gp))
print('closest latitude: ', ds1.lat[idx_lat].values)
print('closest longitude: ', ds1.lon[idx_lon].values)
#latitude = 30.50; longitude = 90.75 ### Zhadang
latitude = float(ds1.lat[idx_lat].values)
longitude = float(ds1.lon[idx_lon].values)

# Load data
era5 = xr.open_dataset(era5_file)
print(era5.time.values[-1])
era5_static = xr.open_dataset(era5_static_file)

# Select grid point
era5 = era5.sel(lat=latitude, lon=longitude)
era5_static = era5_static.sel(lat=latitude,lon=longitude)

###select only time slice
#breakpoint()

era5 = era5.sel(time=slice(time_start,time_end))
print(era5.time.values[0])
print(era5.time.values[-1])
print('First timestamp: ', era5.time[0].values, ' last timestamp: ', era5.time[-1].values)
print("Altitude of gridpoint ", era5_static.z.values/g)         # Aus Geopotentialhöhe Geländehöhe berechnen

# calculate height difference
height_diff = target_altitude  - era5_static.z.values/g         # Target Ausgangspunkt der Lapse rates (hier die Höhe der AWS)
print(height_diff)
time = era5['time'].to_index()                              # era5['time'] ist das Gleiche wie era5.time
time_local = time+pd.Timedelta(hours=timezone_difference_to_UTC)    # auf UTC Zeitverschiebung draufrechnen (Zeitzone anpassen)


##### calculate variables
temperature = (era5['t2m'].values) + height_diff * lapse_rate_temperature

if lapse_rate_temperature == 0.00:          # == dasselbe wie is
  air_pressure = era5['sp'].values/100
else:
  air_pressure = ((era5['sp'].values)/100) * (1-abs(lapse_rate_temperature)*height_diff/era5['t2m'].values) ** ((g*M)/(R*abs(lapse_rate_temperature)))          # Barometrische Höhenformel beinhaltet Temperaturgradienten.
cloud_cover = era5['tcc'].values
ws10 = xr.ufuncs.sqrt(era5['v10'].values**2 + era5['u10'].values**2)
windspeed = ws10 * ((np.log(2/z0))/np.log(10/z0)) * scaling_wind
es_d2m = teten_a1 * np.exp(teten_a3 * (era5['d2m'].values-zero_temperature)/(era5['d2m'].values-teten_a4))      #Saettigungsdampfdruck berechnen
es_t2m = teten_a1 * np.exp(teten_a3 * (era5['t2m'].values-zero_temperature)/(era5['t2m'].values-teten_a4))      #Saettigungsdampfdruck berechnen
relative_humidity = 100 * (es_d2m)/(es_t2m) + height_diff * lapse_rate_relative_humidity

snowfall = (era5['sf'].values.flatten() + height_diff * lapse_rate_snowfall) * (ice_density/density_fresh_snow)
### or if multidimensional # relative_humidity[relative_humidity>100]=100
total_precipitation = era5['tp'].values.flatten() * 1000 + height_diff * lapse_rate_total_precipitation         ### convert from m to mm
shortwave_in = (era5['ssrd'].values/(60*60)).flatten()                                                          ### convert from J/m^(-2) to Wm^(-2)
longwave_in = (era5['strd'].values/(60*60)).flatten()                                                          ### convert from J/m^(-2) to Wm^(-2)

def remove_nan(var):
  var = var[~np.isnan(var)]
  return var

idx = np.isnan(total_precipitation)

#total_precipitation = remove_nan(total_precipitation)
#snowfall = remove_nan(snowfall)
#shortwave_in = remove_nan(shortwave_in)

relative_humidity[relative_humidity>100]=100
shortwave_in[shortwave_in<0]=0
longwave_in[longwave_in<0]=0
total_precipitation[total_precipitation<0]=0
snowfall[snowfall<0]=0
#plt.plot(total_precipitation[0:24])
#plt.show()

print('T2')
mmm_nan(temperature)
mmm_nan(era5['t2m'].values)

print('PRES')
mmm_nan(era5.sp)
mmm_nan(air_pressure)

print('N')
mmm_nan(cloud_cover)

print('U2')
mmm_nan(windspeed)

print('RH2')
mmm_nan(relative_humidity)

print('TP')
mmm_nan(total_precipitation)

print('SNOWFALL')
mmm_nan(snowfall)

print('G')
mmm_nan(shortwave_in)

print('LWin')
mmm_nan(longwave_in)

check(temperature,'T2',316.16,230.16)
check(air_pressure,'PRES',700.0,500.0)
check(cloud_cover,'N',1.0,0.0)
check(windspeed,'U2',50.0,0.0)
check(relative_humidity,'RH2',100.0,0.0)
check(total_precipitation,'TP',20.0,0.0)
check(snowfall,'SNOWFALL',0.1,0.0)
check(shortwave_in,'G',1600.0,0.0)
check(longwave_in,'LWin',400.0,0.0)

## COSIPY or PYPDD:
    # COSIPY
raw_data_cosipy = {'TIMESTAMP': time_local,
            'T2': temperature,
            'PRES': air_pressure,
            'N': cloud_cover,
            'U2': windspeed,
            'RH2':relative_humidity,
            'RRR': total_precipitation,
            'SNOWFALL': snowfall,
            'G': shortwave_in
            }
df = pd.DataFrame(raw_data_cosipy, columns = ['TIMESTAMP', 'T2', 'PRES', 'N', 'U2', 'RH2', 'RRR', 'SNOWFALL', 'G', 'LWin'])
df.to_csv(output_cosipy,index=False)

    # PYPDD
raw_data_pypdd = {'TIMESTAMP': time_local,
            'temp': temperature,
            'prec': total_precipitation,
            }
df = pd.DataFrame(raw_data_pypdd, columns = ['TIMESTAMP', 'temp', 'prec'])

# Needed: two arrays ``temp`` and ``prec`` of shape ``(t, x, y)``

df.to_csv(output_pypdd,index=False)