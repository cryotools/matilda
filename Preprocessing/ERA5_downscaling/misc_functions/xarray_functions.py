import pandas as pd

def insert_var(ds, var, name, units, long_name):
    ds[name] = (('time'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def insert_var_2D(ds, var, name, units, long_name):
    ds[name] = (('time', 'lat', 'lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name
    return ds

def insert_var_static(ds, var, name, units, FillValue, missing_value, long_name):
    ds[name] = (('lat' , 'lon'), var)
    ds[name].attrs['units'] = units
    ds[name].attrs['long_name'] = long_name

def string_to_numeric(df,var):
    df[var] = df[var].apply(pd.to_numeric, errors='coerce')

def return_value_pit(var,point):
    var = var.sel(time=slice(point,point))
    print(point,var.values)
    return var

def convert_cumulative_to_diff(var):
    vardiff = xr.DataArray(np.diff(var))
    return vardiff


