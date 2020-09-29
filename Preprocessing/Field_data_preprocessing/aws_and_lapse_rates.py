import numpy as np; import pandas as pd; import xarray as xr; import salem
from pathlib import Path
import matplotlib.pyplot as plt
import sys; sys.path.append('/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Preprocessing/ERA5_downscaling/')
from fundamental_physical_constants import g, M, R

home = str(Path.home())
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'

##

aws_up = pd.read_csv(working_directory + 'Minikin/Cognac_glacier/80_2019_09_12_refin.csv', sep=';', decimal=',',
                     usecols=range(0, 4))
aws_up.columns = ['datetime', 'G', 'temp', 'hum']
aws_up.datetime = pd.to_datetime(aws_up.datetime)
aws_up.set_index(aws_up.datetime, inplace=True)
aws_up = aws_up.drop(['datetime'], axis=1)
aws_up = aws_up.shift(-2, axis=0)  # Only -2h timeshift results in the correct curves. BUT WHY????
aws_up.temp = aws_up.temp + 273.15
aws_up = aws_up[time_start: time_end]

##
def sdss_open(path, varname = 'value', celsius = False, resample = True,
              resample_rate = 'H', resample_method = 'mean', time_slice = False, time_start = None, time_end = None):
    aws = pd.read_csv(path)
    aws.columns = ['datetime', varname]
    aws.set_index(pd.to_datetime(aws.datetime), inplace=True)
    aws = aws.drop(['datetime'], axis=1)
    if celsius:
        aws[varname] = aws[varname] + 273.15
    if resample and resample_method == 'mean':
        aws = aws.resample(resample_rate).mean()
    elif resample and resample_method == 'sum':
        aws = aws.resample(resample_rate).sum()

    if time_slice and time_start is None and time_end is None:
        print("******************************************************************************************")
        print("WARNING! No time slice defined. Please set valid arguments for time_start and/or time_end.")
        print("******************************************************************************************")
    elif time_slice:
        aws = aws[time_start: time_end]
    return aws

time_start = '2018-09-07 18:00:00'  # longest timeseries of waterlevel
time_end = '2019-09-14 09:00:00'
path = working_directory + 'AWS_atbs/temp_2017-05-30_2020-09-16.csv'

aws_down2 = sdss_open(path, varname='temp', celsius=True, time_slice=True, time_start=time_start, time_end=time_end)

##
alt_hobo = 3342
lat_hobo = 41.068228
lon_hobo = 75.99092
alt_aws_far = 3023
alt_aws_up = 3894
alt_aws_down = 2250
lapseT = -(aws_down.temp.mean() - aws_up.temp.mean()) / (alt_aws_down - alt_aws_up)  # literature: -0.006 K/m
