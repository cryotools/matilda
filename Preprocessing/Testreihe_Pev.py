##
from pathlib import Path; home = str(Path.home())      ### Zieht sich home vom system
#import sys; sys.path.append(home + '/Seafile/Ana-Lena_Phillip/scripts/')
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##
# File organization
working_directory = '/Seafile/Ana-Lena_Phillip/data/input_output/'

era5_static_file = home + working_directory + 'ERA5/global/ERA5_land_Z_geopotential.nc'

# hourly values
era5_file_pev = home + working_directory + "/ERA5/Tien-Shan/Tien-Shan_ERA5Land_pev_2016-18_inv.nc"
era5_file_temp = home + working_directory + "ERA5/Tien-Shan/Tien-Shan_ERA5Land_t2m_2016-18.nc"

#Daily values (mean temp and sum of pev which is inverted)
era5_daily_file = home + working_directory + "ERA5/Tien-Shan/Tien-Shan_ERA5Land_pev-t2m-daily_2016-18.nc"

# Load data
era5_static = xr.open_dataset(era5_static_file)
era5_pev = xr.open_dataset(era5_file_pev)
era5_temp = xr.open_dataset(era5_file_temp)
era5_daily = xr.open_dataset(era5_daily_file)

##
# Downscaling Lapse rate pev calculation
pev_mean = era5_daily.pev # PEV is inverted with cdo tools before calculating the daily sum
pev_mean = pev_mean.mean(dim="time")
pev_mean = pev_mean.to_dataframe()
pev_mean.reset_index(inplace=True)

longitude = era5_daily.lon; latitude = era5_daily.lat   # Area for Pev Downscaling
era5_static = era5_static.sel(lat=latitude, lon=longitude, method="nearest")
era5_static = era5_static.z
era5_static = era5_static.to_dataframe()
era5_static.reset_index(inplace=True)
era5_static["elevation"] = era5_static["z"]/9.80665
era5_static = era5_static[["lat", "lon", "elevation"]]

df_downscaling = pd.merge(pev_mean, era5_static, right_index=True, left_index=True)
df_downscaling = df_downscaling[["lat_x", "lon_x", "pev", "elevation"]]
df_downscaling.columns = ["lat", "lon", "pev", "elevation"]
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

##
# plotting the actual points as scatter plot
plt.scatter(x, y, color="m",
            marker="o", s=30)

# predicted response vector
y_pred = b_0 + b_1 * x

# plotting the regression line
plt.plot(x, y_pred, color="g")

# putting labels
plt.xlabel('Elevation [m]')
plt.ylabel("Potential evaporation [m]")
plt.show()

##
# Downscaling
lapse_rate_pev = b_1
lapse_rate_temperature = -0.006             # K/m  temperature lapse rate

reference_heights = np.arange(500, 5500, 1000)

## daily values, pev sum & t2m mean calculated and merged with cdo
#reopen elevation data
elevation = xr.open_dataset(era5_static_file)
elevation = elevation.sel(lat=latitude, lon=longitude, method="nearest")

elevation_array = elevation.to_array(dim="z")
pev_array = era5_daily.pev
temp_array = era5_daily.t2m

elevation_array = elevation_array/9.80665

# calculation of all the height differences (elevation and reference height) per cell
height_diff = {}
for i in reference_heights:
    height_diff[i] = abs(elevation_array.values - i)

# Downscaling of PEV for each reference height
pev_array_down = {}
for i in height_diff.keys():
    pev_array_down[i] = (pev_array + height_diff[i] * lapse_rate_pev) * 1000

for i in pev_array_down.keys():
    pev_array_down[i] = np.where(pev_array_down[i] < 0, 0, pev_array_down[i])

# Downscaling of temp for each reference height
temp_array_down = {}
for i in height_diff.keys():
    temp_array_down[i] = (temp_array + height_diff[i] * lapse_rate_temperature) - 273.15

# Calculation of PEV with the downscaled temp with the formula of Oudin et al.
# unit is mm / day
solar_constant = (1376 / 1000000) * 86400 # from 1376 J/m2s to MJm2d
extra_rad = 27.086217947590317
latent_heat_flux = 2.45
water_density = 1000

pev_calc_array={}
for i in temp_array_down.keys():
    pev_calc_array[i] = np.where(temp_array_down[i] + 5 > 0, ((extra_rad / (water_density * latent_heat_flux)) * ((temp_array_down[i] + 5)/100)*1000), 0)

# Differences between PEV downscaled and PEV calculated
diff_array = {}
for i, j in zip(pev_array_down.keys(), pev_calc_array.keys()):
    diff_array[i] = abs(pev_array_down[i] - pev_calc_array[j])

# Mean difference over all timesteps
diff_mean_array = {}
for i in diff_array.keys():
    diff_mean_array[i] = np.mean(diff_array[i], axis=0)

for i in diff_mean_array.keys():
    print("Maximum difference of " + str(i) + "m reference height is " + str(float(np.max(diff_mean_array[i]))))
    print("Minimum difference of " + str(i) + "m reference height is " + str(float(np.min(diff_mean_array[i]))))
    print("Mean difference of " + str(i) + "m reference height is " + str(float(np.mean(diff_mean_array[i]))))
