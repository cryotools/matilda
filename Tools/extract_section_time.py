import xarray as xr
ds = xr.open_dataset('best_cosipy_output_no1_2000-20.nc')
ds = ds.sel(time=slice('20110101T00:00', '20181231T23:00'))
ds.to_netcdf('test.nc')
