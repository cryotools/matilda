##
from orographic_precipitation import compute_orographic_precip
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path; home = str(Path.home())      ### Zieht sich home vom system
working_directory = home + '/Seafile/Ana-Lena_Phillip/data/matilda/pypdd'
input_path = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/'
static_path = home + '/Seafile/Ana-Lena_Phillip/data/input_output/static/'

input_file = '20200625_Umrumqi_ERA5_2011_2018_cosipy.nc'
static_file = 'Urumqi_static.nc'

input = input_path + input_file
static = static_path + static_file

#Time slice:
time_start = '2011-01-01T00:00'
time_end = '2018-12-31T23:00'

help(compute_orographic_precip)
## Data
DS = xr.open_dataset(input)
DS = DS.sel(time=slice(time_start, time_end))
static_ds = xr.open_dataset(static)

elevation = DS.HGT.values
dx = 333 # horizontal and vertical resolution in meters (0.03 degree)
dy = 333 # the way I understood it, it's the grid size? size of each pixel?

## orographic prec function example
# isolated circular Gaussian hill
def gauss_topography(dx, dy):
  """Returns synthetic topography for testing.
  Analogous to Fig 4 in Smith and Barstedt, 2004.
  """
  h_max = 500.
  x0 = -25e3
  y0 = 0
  sigma_x = sigma_y = 15e3

  x, y = np.arange(-100e3, 200e3, dx), np.arange(-150e3, 150e3, dy)
  X, Y = np.meshgrid(x, y) # grid with steps of dx/dy which is the resolution

  h = (h_max *
       np.exp(-(((X - x0)**2 / (2 * sigma_x**2)) +
                ((Y - y0)**2 / (2 * sigma_y**2))))) # calculation of the elevation in this grid

  return X, Y, h

def plot2d(X, Y, field):
  """Function that plots precipitation matrices"""
  fig, ax = plt.subplots(figsize=(6, 6))
  pc = ax.pcolormesh(X, Y, field)
  ax.set_aspect(1)
  fig.colorbar(pc) # plotting a "grid" with elevation as the field input

dx_2 = 750.
dy_2 = 750.
X, Y, elevation_2 = gauss_topography(dx_2, dy_2) # creating the gaussian hill with the resolution

plot2d(X, Y, elevation_2)
plt.show()


# we need an array with elevation in this grid and two resolutions
## orographic prec
# lapse rate
prec = DS.RRR.mean(dim="time")
# number of observations/points
n = np.size(elevation)
# mean of x and y vector
m_x, m_y = np.mean(elevation), np.mean(prec)
# calculating cross-deviation and deviation about x
SS_xy = np.sum(prec * elevation) - n * m_y * m_x
SS_xx = np.sum(elevation * elevation) - n * m_x * m_x
# calculating regression coefficients
b_1 = SS_xy / SS_xx


lapse_rate = -5.8
lapse_rate_m = -6.5 # standard value
ref_density = 7.4e-3

param = {
'latitude': 43.11,
'precip_base': 0.07,               # uniform precipitation rate [mm hr-1]
'wind_speed': 1.9,                 # [m s-1]
'wind_dir': 270,                   # wind direction (270: west)
'conv_time': 1000,                 # cloud water to hydrometer conversion time
'fall_time': 1000,                 # hydrometer fallout time
'nm': 0.005,                       # moist stability frequency [s-1]
'hw': 5000,                        # water vapor scale height [m]
'cw': ref_density * lapse_rate_m / lapse_rate   # uplift sensitivity [kg m-3]
      # product of saturation water vapor sensitivity ref_density [kg m-3] and environmental lapse rate (lapse_rate_m / lapse_rate)
}

P_2 = compute_orographic_precip(elevation_2, dx_2, dy_2, **param)
plot2d(X, Y, P_2)
plt.show()

# our values
P = compute_orographic_precip(elevation, dx, dy, **param) # array with same size as elevation, precipitation rate mm [hr]
plt.imshow(elevation)
plt.colorbar()
plt.show()
# returns a grid
