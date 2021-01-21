import numpy as np
import xarray as xr
import salem
from pathlib import Path
import sys
home = str(Path.home())
working_directory = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/'

in_file = working_directory + 'ssrd_no182_ERA5_Land_1981_2019.nc'
in_file2 = working_directory + 't2m_no182_ERA5_Land_1981_2019.nc'
# in_file81_ss = working_directory + 'ERA5_land_HMA_Region1_ssrd_1981.nc'
# in_file19_ss = working_directory + 'ERA5_land_HMA_Region1_ssrd_2019.nc'
# in_file81_t = working_directory + 'ERA5_land_HMA_Region1_t2m_1981.nc'
# in_file19_t = working_directory + 'ERA5_land_HMA_Region1_t2m_2019.nc'

in_file = working_directory + 'no182ERA5_Land_1981_2019.nc'

ds = xr.open_dataset(in_file)
# ds2 = xr.open_dataset(in_file2)

# ds81_ss = xr.open_dataset(in_file81_ss)
# ds19_ss = xr.open_dataset(in_file19_ss)
#
# ds81_t = xr.open_dataset(in_file81_t)
# ds19_t = xr.open_dataset(in_file19_t)

pick = ds.sel(lat=41.0, lon=75.9, method='nearest')

pick_df = pick.to_dataframe()
pick_df.to_csv(home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182ERA5_Land_1981_2019.csv')
