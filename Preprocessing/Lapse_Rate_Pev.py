##
from pathlib import Path; home = str(Path.home())
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

##
ncfile_z = home + "/Seafile/Ana-Lena_Phillip/data/input_output/ERA5/global/ERA5_land_Z_geopotential.nc"
ncfile_evap = home + "/Seafile/Ana-Lena_Phillip/data/input_output/ERA5/202004_Urumqi_era5_evap_run_2011-2018.nc"

nc_z = xr.open_dataset(ncfile_z)
nc_evap = xr.open_dataset(ncfile_evap)

pev = nc_evap.pev
pev = pev*-1

pev_mean = pev.mean(dim="time")
pev_df = pev_mean.to_dataframe()
pev_df.reset_index(inplace=True)

longitude = pev.longitude
latitude = pev.latitude
lat_urumqi = 43.00
lon_urumqi = 86.75

nc_z_subset = nc_z.sel(lat=latitude, lon=longitude, method="nearest")
z = nc_z_subset.z
z_df = z.to_dataframe()
z_df.reset_index(inplace=True)
z_df["elevation"] = z_df["z"]/9.80665

z_urumqi = nc_z.z.sel(lat=lat_urumqi, lon=lon_urumqi, method="nearest")
elevation_urumqi = z_urumqi/9.80665

df = pd.merge(pev_df, z_df)
df = df[["latitude", "longitude", "pev", "elevation"]]
y = df.pev.values
x = df.elevation.values


##
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


##
lapse_rate = b_1
elev_glac = 4025                                # Reference elevation of AWS
h_diff = abs(elev_glac-float(elevation_urumqi))
pev_uru = -nc_evap.pev.sel(latitude=lat_urumqi, longitude=lon_urumqi, method="nearest")     # PEV of closest GP inverted
pev_df = pev_uru.to_dataframe()["pev"]
pev_df_down = pev_df + (lapse_rate * h_diff)        # Series object
pev_df_down = pev_df_down.reset_index()        # pandas.core.frame.DataFrame , time is datetime64[ns]
pev_df_down.set_index('time', inplace=True)     # DataFrame with DateTime-Index
pev_df_down[pev_df_down < 0] = 0                # Setting all values <0 to 0
pev_df_down = pev_df_down*1000                  # m in mm
pev_df_down.plot.line()
plt.ylabel("Potential evaporation [mm]")
plt.show()
# plt.savefig(home + '/Seafile/Ana-Lena_Phillip/data/figures/pev_hourly_no1.png')

# Resample:
pev_daily = pev_df_down.resample('D').sum()
pev_daily.plot.line()
plt.show()
pev_daily.describe()

pev_monthly = pev_df_down.resample('M').sum()
pev_monthly.plot.line()
plt.show()
pev_monthly.describe()

pev_yearly = pev_df_down.resample('Y').sum()
pev_yearly.plot.line()
plt.show()
pev_yearly.describe()
