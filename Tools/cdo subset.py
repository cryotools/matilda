from cdo import *
cdo = Cdo()
cdo.debug = True

working_directory = '/Seafile/Ana-Lena_Phillip/data/input_output/'
in_file = home + working_directory + 'ERA5/No1_Urumqi_ERA5_2000_201907.nc'
out_file = home + working_directory + 'ERA5/No1_Urumqi_ERA5_2011_2018.nc'

date1 = '2011-01-01T00:00:00'
date2 = '2018-12-31T23:00:00'

cdo.seldate(date1+','+date2, input = in_file, output = out_file, options ='-f nc')
