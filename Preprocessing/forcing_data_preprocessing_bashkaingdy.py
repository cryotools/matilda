##
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import salem
from pathlib import Path
import sys
import socket
import os
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
import Downscaling.scikit_downscale_matilda as sds
from Preprocessing_functions import pce_correct



# interactive plotting?
# plt.ion()

##########################
#   Data preparation:    #
##########################

## ERA5 closest gridpoint:

# Apply '/Ana-Lena_Phillip/data/matilda/Tools/ERA5_Subset_Routine.sh' for ncdf-subsetting

in_file = home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/new_grib_conversion' +\
          '/t2m_tp_no182_ERA5L_1982_2020.nc'
ds = xr.open_dataset(in_file)
pick = ds.sel(latitude=41.134066, longitude=75.942381, method='nearest')           # closest to AWS location
era = pick.to_dataframe().filter(['t2m', 'tp'])
# era = era.tz_localize('UTC')

total_precipitation = np.append(0, (era.drop(columns='t2m').diff(axis=0).values.flatten()[1:]))   # transform from cumulative values
total_precipitation[total_precipitation < 0] = era.tp.values[total_precipitation < 0]
era['tp'] = total_precipitation

era['tp'][era['tp'] < 0.00004] = 0         # Refer to https://confluence.ecmwf.int/display/UDOC/Why+are+there+sometimes+small+negative+precipitation+accumulations+-+ecCodes+GRIB+FAQ
era['tp'] = era['tp']*1000                 # Unit to mm

# era.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/t2m_tp_ERA5L_no182_41.1_75.9_1982_2019.csv')

era_D = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})


## AWS Bash Kaingdy:
aws_full = pd.read_csv(home + '/EBA-CA/Tianshan_data/AWS_atbs/download/New/atbs_2017-2021.csv',
                 parse_dates=['time'], index_col='time')
aws_full.t2m = aws_full.t2m + 273.15
aws_full = aws_full.resample('H').agg({'t2m': 'mean', 'tp': 'sum', 'ws': 'mean', 'wd': 'mean', 'rh': 'mean'})
aws_full['tp'][aws_full['tp'] > 20] = 0     # Three most extreme outliers within 3hrs. Due to AWS maintenance?

aws = aws_full.drop(columns=['wd', 'rh'])

    # Application of transfer function to account for solid precipitation undercatch (Kochendorfer et.al. 2020)
aws['tp'] = pce_correct(aws['ws'], aws['t2m'], aws['tp'])

# aws['tp'][slice('2017-06-03', '2019-12-31')].describe()
# aws['tp'][slice('2017-06-03', '2019-12-31')].sum()
# 666.9654912649049/620.2                                       # Correction increased the tp by 7.5% (942 days)

    # Downscaling cannot cope with data gaps:                   But BCSD CAN!!!!!!
aws_D = aws.resample('D').agg({'t2m': 'mean', 'tp': 'sum', 'ws': 'mean'})
aws_D_int = aws_D.interpolate(method='spline', order=2)           # No larger data gaps after 2017-07-04

aws_D[slice('2017-07-14', '2021-06-06')].isna().sum()
aws_D_int.isna().sum()

aws_D[slice('2017-07-14', '2021-06-06')][pd.isnull(aws_D.t2m)]

# aws[slice('2020-01-01', '2020-12-31')]['tp'] = np.NaN
# aws.to_csv(home + '/EBA-CA/Tianshan_data/AWS_atbs/' + 'aws_preprocessed_2017-06_2021-05.csv')

#     # Downscaling cannot cope with data gaps:                   But BCSD CAN!!!!!!
# aws_D = aws.resample('D').agg({'t2m': 'mean', 'tp': 'sum', 'ws': 'mean'})
# aws_D_int = aws_D.interpolate(method='spline', order=2)           # No larger data gaps after 2017-07-04


## Minikin-data:
# minikin = pd.read_csv(home + "/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/Old" +
#                              "/Bash-Kaindy_preprocessed_forcing_data.csv",
#                       parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
# minikin = minikin.filter(like='_minikin')
# minikin.columns = ['t2m']
# minikin = minikin.resample('D').mean()


## CMIP6 data Bash Kaingdy:
cmip = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/' +
                       'CMIP6_mean_41-75.9_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip = cmip.filter(like='_45')            # To select scenario e.g. RCP4.5 from the model means
# cmip = cmip.tz_localize('UTC')
cmip.columns = era.columns
cmip = cmip.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})       # Already daily but wrong daytime (12:00:00 --> lesser days overall).
cmip = cmip.interpolate(method='spline', order=2)       # Only 25 days in 100 years, only 3 in fitting period.


## Overview

# AWS location: 41.134066, 75.942381
# aws:      2017-07-14 to 2021-06-06    --> Available from 2017-06-02 but biggest datagap: 2017-07-04 to 2017-07-14
# era:      1981-01-01 to 2020-12-31
# minikin:  2018-09-07 to 2019-09-13
# cmip:     2000-01-01 to 2100-12-30

# t = slice('2018-09-07', '2019-09-13')
# d = {'AWS': aws[t]['t2m'], 'ERA5': era[t]['t2m'], 'Minikin': minikin[t]['t2m'], 'CMIP6': cmip[t]['t2m']}
# data = pd.DataFrame(d)
# data.plot(figsize=(12, 6))
#
# t = slice('2017-07-14', '2021-06-06')
# d = {'AWS': aws_D[t]['tp'], 'ERA5': era_D[t]['tp'], 'CMIP6': cmip[t]['tp']}
# data = pd.DataFrame(d)
# data.plot(figsize=(12, 6))
# # data.describe()
# plt.show()
# data.sum()
#
# t = slice('2018-09-07', '2019-09-13')
# d = {'ERA5': era[t]['tp'], 'AWS': aws[t]['tp']}
# data = pd.DataFrame(d)
# data.plot(figsize=(12, 6))
# # data.describe()
# plt.show()
