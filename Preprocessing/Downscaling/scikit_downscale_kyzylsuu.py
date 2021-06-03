##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as stats
from pathlib import Path
import sys
import socket
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/scripts/Preprocessing/Downscaling'
import os
os.chdir(wd)
sys.path.append(wd)
from sklearn.linear_model import LinearRegression
from skdownscale.pointwise_models import BcsdTemperature
import scikit_downscale_matilda as sds


# interactive plotting?
# plt.ion()

##########################
#   Data preparation:    #
##########################

temp = pd.read_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/temp_kyzylsuu_2007-2015.csv',
                   parse_dates=['time'], index_col='time')
prec = pd.read_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/prec_kyzylsuu_2007-2014.csv',
                   parse_dates=['time'], index_col='time')

met = pd.merge(temp, prec, how='outer', left_index=True, right_index=True)
met.to_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/met_data_full_kyzylsuu_2007-2015.csv')























# ERA5 closest gridpoint:
era = pd.read_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy'
                           '/no182ERA5_Land_lat41.0_lon75.9_alt3839.6_1981-2019.csv', index_col='time')
era.index = pd.to_datetime(era.index, format='%d.%m.%Y %H:%M')
era = era.tz_localize('UTC')
era = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})


# AWS Bash Kaingdy:
aws = pd.read_csv(home + '/EBA-CA/Tianshan_data/AWS_atbs/atbs_met-data_2017-2020.csv',
                  parse_dates=['datetime'], index_col='datetime')
aws = aws.shift(periods=6, freq="H")                                     # Data is still not aligned with UTC
aws = aws.tz_convert('UTC')
aws = aws.drop(columns=['rh', 'ws', 'wd'])                    # Need to be dataframes not series!
aws.columns = era.columns
    # Downscaling cannot cope with data gaps:
aws = aws.resample('D').agg({'t2m': 'mean', 'tp':'sum'})
aws = aws.interpolate(method='spline', order=2)           # No larger data gaps after 2017-07-04


# Minikin-data:
minikin = pd.read_csv(home + "/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy" +
                             "/Bash-Kaindy_preprocessed_forcing_data.csv",
                      parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
minikin = minikin.filter(like='_minikin')
minikin.columns = ['t2m']
minikin = minikin.resample('D').mean()


# CMIP6 data Bash Kaingdy:
cmip = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/' +
                       'ssp2_4_5_41-75.9_2000-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip = cmip.filter(like='_mean')
cmip = cmip.tz_localize('UTC')
cmip.columns = era.columns
cmip = cmip.resample('D').mean()        # Already daily but wrong daytime (12:00:00 --> lesser days overall).
cmip = cmip.interpolate(method='spline', order=2)       # Only 25 days in 100 years, only 3 in fitting period.


## Overview

# aws:      2017-07-14 to 2020-09-30    --> Available from 2017-06-02 but biggest datagap: 2017-07-04 to 2017-07-14
# era:      1981-01-01 to 2019-12-31
# minikin:  2018-09-07 to 2019-09-13
# cmip:     2000-01-01 to 2100-12-30

# t = slice('2018-09-07', '2019-09-13')
# d = {'AWS': aws[t]['t2m'], 'ERA5': era[t]['t2m'], 'Minikin': minikin[t]['t2m'], 'CMIP6': cmip[t]['t2m']}
# data = pd.DataFrame(d)
# data.plot(figsize=(12, 6))




#################################
#    Downscaling temperature    #
#################################
