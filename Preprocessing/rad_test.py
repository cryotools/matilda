import pandas as pd
import numpy as np
import xarray as xr

# source: my code from sandbox/WFDEI_to_dataframe.ipynb
path_to_scheme = "/home/phillip/Seafile/Ana-Lena_Phillip/PycharmProjects/Ana/pur_scheme.csv"
# convert our basin scheme to simple lists
def schema_to_lists(path):
    schema = pd.read_csv(path, usecols=[0, 1])
    lat = schema.Y.values
    lon = schema.X.values
    return lat, lon

# read our WFDEI dataset and cutting it through our scheme coordinates
# xarray dataframe as output
def data_reading_cutting(path, lat, lon):
    data = xr.open_dataset(path, decode_times=False)
    data['time'] = pd.date_range('1979-01-01', '2014-12-31', name='time')
    return data.sel_points(lat = lat, lon = lon)

# convert our xarray dataframe to simple pandas dataframe
def xdata_to_pdata(xdata, var_name):
    pdata = pd.DataFrame(index=pd.date_range('1979-01-01', '2014-12-31', name='Date'))
    for i in range(xdata.dims['points']):
        pdata[var_name+'_'+str(i)] = xdata[var_name][i].data
    return pdata

# function for potential evaporation (PET) by (Oudin et al., 2005)
def PET(data, path_to_scheme):
    # Reference: http://www.fao.org/docrep/x0490e/x0490e07.htm
    # use with caution for latitudes out of range 0-67 degrees

    # Part 1. Avarage latitude calculation
    # read watershed scheme
    schema = pd.read_csv(path_to_scheme, usecols=[0, 1])
    # calculate mean watershed latitude
    # and convert it from degrees to radians
    lat = np.deg2rad(schema.Y.values.mean())

    # Part 2. Extraterrrestrial radiation calculation
    # set solar constant (in W m-2)
    Rsc = 1367
    # calculate day of the year array
    doy = np.array([i for i in range(1, 367)])
    # calculate solar declination dt (in radians)
    dt = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
    # calculate sunset hour angle (in radians)
    ws = np.arccos(-np.tan(lat) * np.tan(dt))
    # Calculate sunshine duration N (in hours)
    N = 24 / np.pi * ws
    # Calculate day angle j (in radians)
    j = 2 * np.pi / 365.25 * doy
    # Calculate relative distance to sun
    dr = 1.0 + 0.03344 * np.cos(j - 0.048869)
    # Calculate extraterrestrial radiation (J m-2 day-1)
    Re = Rsc * 86400 / np.pi * dr * (ws * np.sin(lat) * np.sin(dt)\
           + np.sin(ws) * np.cos(lat) * np.cos(dt))
    # convert from J m-2 day-1 to MJ m-2 day-1
    Re = Re/10**6
s with