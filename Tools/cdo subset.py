from cdo import *
cdo = Cdo()
cdo.debug = True

working_directory = 'sftp://cirr/data/projects/ebaca/No1_COSIPY'
in_file = working_directory + 'best_cosipy_output_no1_2000-20.nc'
out_file = working_directory + 'best_cosipy_output_no1_2011-18.nc'

date1 = '2011-01-01T00:00:00'
date2 = '2018-12-31T23:00:00'

cdo.seldate(date1+','+date2, input = in_file, output = out_file, options ='-f nc')
