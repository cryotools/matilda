import xarray as xr
input = '/data/projects/prime-SG/ERA5Land/Tien_Shan_Region/20200921_ERA5-Land_Tien_Shan_Region_2001_2019.nc'
# input = '/data/projects/ebaca/data/input_output/input/ERA5/20200722_Karabatkak_ERA5L_1982_2019.nc'
output_folder = '/data/projects/ebaca/data/input_output/input/ERA5/'
output_file = '20200722_Karabatkak_ERA5L_1982_83.nc'

ds = xr.open_dataset(input)
ds = ds.sel(time=slice('20010101T00:00', '20021231T23:00'))
ds.to_netcdf(output_folder + output_file)


