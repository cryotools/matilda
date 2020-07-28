import xarray as xr

ds = xr.open_dataset('Zhadang_ERA5_2009.nc')
T2 = ds.T2.values
breakpoint()
