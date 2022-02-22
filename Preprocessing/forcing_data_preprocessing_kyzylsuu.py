##
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import socket
import os
from bias_correction import BiasCorrection

warnings.filterwarnings("ignore")  # sklearn
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/matilda/Preprocessing'
os.chdir(wd + '/Downscaling')
sys.path.append(wd)

##########################
#   Data preparation:    #
##########################

## AWS Chong Kyzylsuu:
aws_temp = pd.read_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/' +
                       'temp_kyzylsuu_2007-2015.csv',
                       parse_dates=['time'], index_col='time')
aws_temp['t2m'] = aws_temp['t2m'] + 273.15
aws_temp_D = aws_temp.resample('D').mean()                # Precipitation data is daily!

aws_prec = pd.read_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/' +
                       'prec_kyzylsuu_2007-2014.csv',
                       parse_dates=['time'], index_col='time')

# Interpolate shorter data gaps and exclude the larger one to avoid NAs:

# temp data gaps: 2010-03-30' to '2010-04-01', '2011-10-12' to '2011-10-31', '2014-05-02' to '2014-05-03',
# '2015-06-05' to '2015-06-06'
aws_temp_D_int1 = aws_temp_D[slice('2007-08-10', '2011-10-11')]
aws_temp_D_int2 = aws_temp_D[slice('2011-11-01', '2016-01-01')]
aws_temp_D_int1 = aws_temp_D_int1.interpolate(method='spline', order=2)
aws_temp_D_int2 = aws_temp_D_int2.interpolate(method='spline', order=2)
aws_temp_D_int = pd.concat([aws_temp_D_int1, aws_temp_D_int2], axis=0)      # Data gap of 18 days in October 2011


aws = pd.merge(aws_temp_D, aws_prec, how='outer', left_index=True, right_index=True)
# aws.to_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/' +
# 'met_data_full_kyzylsuu_2007-2015.csv')


## ERA5L Gridpoint:

# Apply '/Ana-Lena_Phillip/data/matilda/Tools/ERA5_Subset_Routine.sh' for ncdf-subsetting

in_file = home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/t2m_tp_kysylsuu_ERA5L_1982_2020.nc'
ds = xr.open_dataset(in_file)
pick = ds.sel(latitude=42.191433, longitude=78.200253, method='nearest')           # closest to AWS location
# pick = pick.sel(time=slice('1989-01-01', '2019-12-31'))                      # start of gauging till end of file
era = pick.to_dataframe().filter(['t2m', 'tp'])

total_precipitation = np.append(0, (era.drop(columns='t2m').diff(axis=0).values.flatten()[1:]))   # transform from cumulative values
total_precipitation[total_precipitation < 0] = era.tp.values[total_precipitation < 0]
era['tp'] = total_precipitation

era['tp'][era['tp'] < 0.000004] = 0          # Refer to https://confluence.ecmwf.int/display/UDOC/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ
era['tp'] = era['tp']*1000                   # Unit to mm

era.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/t2m_tp_ERA5L_kyzylsuu_42.2_78.2_1982_2020.csv')

era_D = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})

# Include datagap from AWS to align both datasets
era_temp_D = era_D[['t2m']]
era_temp_D_int1 = era_temp_D[slice('2007-08-10', '2011-10-11')]
era_temp_D_int2 = era_temp_D[slice('2011-11-01', '2016-01-01')]
era_temp_D_int =  pd.concat([era_temp_D_int1, era_temp_D_int2], axis=0)      # Data gap of 18 days in October 2011


era_temp_D.to_csv('/home/phillip/Seafile/Ana-Lena_Phillip/data/input_output/input/downscaling_error/example_scikitdownscale/reanalysis.csv')
aws_temp_D.to_csv('/home/phillip/Seafile/Ana-Lena_Phillip/data/input_output/input/downscaling_error/example_scikitdownscale/obs.csv')


## CMIP6:

cmip = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/' +
                       'CMIP6_mean_42.25-78.25_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip = cmip.filter(like='_45')              # To select scenario e.g. RCP4.5 from the model means
# cmip = cmip.tz_localize('UTC')
cmip.columns = era.columns
cmip = cmip.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})      # Already daily but wrong daytime (12:00:00).
cmip = cmip.interpolate(method='spline', order=2)       # Only 3 days in 100 years, only 3 in fitting period.

cmip[['t2m']].to_csv('/home/phillip/Seafile/Ana-Lena_Phillip/data/input_output/input/downscaling_error/example_scikitdownscale/scenario.csv')


## Overview

# prec: 2007-08-01 to 2014-12-31
# temp: 2007-08-10 to 2016-01-01
# AWS location: 42.191433, 78.200253
# Gauging station Hydromet: 1989-01-01 to 2019-09-09
# CMIP: 2000-01-01 to 2100-12-31
# ERA: 1982-01-01 to 2020-12-31

# t = slice('2007-08-10', '2014-12-31')
# d = {'AWS': aws[t]['t2m'], 'ERA5L': era_D[t]['t2m'], 'CMIP6': cmip[t]['t2m']}
# data = pd.DataFrame(d)
# data.describe()
# data.plot(figsize=(12, 6))
# plt.show()

# t = slice('2007-08-10', '2014-12-31')
# d = {'ERA5': era_D[t]['tp'], 'AWS': aws_prec[t]['tp'], 'CMIP6': cmip[t]['tp']}
# data = pd.DataFrame(d)
# data.resample('M').sum()
# data.resample('M').sum().plot(figsize=(12, 6))
# # data.describe()
# plt.show()


