import numpy as np
import pandas as pd
import xarray as xr
import salem
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import glob

sys.path.append('/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Preprocessing/ERA5_downscaling/')
from fundamental_physical_constants import g, M, R

home = str(Path.home())
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'

## General settings
time_start = '2018-09-07 18:00:00'  # longest timeseries of waterlevel
time_end = '2019-09-14 09:00:00'

alt_hobo = 3342
lat_hobo = 41.068228
lon_hobo = 75.99092
alt_minikin_up = 3894
alt_minikin_down = 2250

## Preprocess Minikin-Combi-Sensors
minikin_up = pd.read_csv(working_directory + 'Minikin/Cognac_glacier/cognac_glacier.csv', sep=';', decimal=',',
                         usecols=range(0, 4))
minikin_up.columns = ['datetime', 'G', 'temp', 'hum']
minikin_up.set_index(pd.to_datetime(minikin_up.datetime), inplace=True)
minikin_up = minikin_up.drop(['datetime'], axis=1)
minikin_up = minikin_up.shift(-2, axis=0)  # Only -2h timeshift results in the correct curves. BUT WHY????
minikin_up.temp = minikin_up.temp + 273.15
minikin_up = minikin_up[time_start: time_end]

minikin_down = pd.read_csv(working_directory + 'Minikin/Bash_Kaindy/80_2019_09_12_refin.csv', sep=';', decimal=',',
                           usecols=range(0, 4))
minikin_down.columns = ['datetime', 'G', 'temp', 'hum']
minikin_down.set_index(pd.to_datetime(minikin_down.datetime), inplace=True)
minikin_down = minikin_down.drop(['datetime'], axis=1)
minikin_down = minikin_down.shift(-2, axis=0)  # Only -2h timeshift results in the correct curves. BUT WHY????
minikin_down.temp = minikin_down.temp + 273.15
minikin_down = minikin_down[time_start: time_end]

## Function to preprocess AWS-Data from SDSS


def sdss_open(path, celsius=False, resample=True, tz_localize = True, timezone = 'Asia/Bishkek',
              resample_rate='H', resample_method='mean', time_slice=False, time_start=None, time_end=None):
    aws = pd.read_csv(path)
    aws.columns.values[0] = 'datetime'
    aws.set_index(pd.to_datetime(aws.datetime), inplace=True)
    aws = aws.drop(['datetime'], axis=1)
    if tz_localize:
        aws = aws.tz_localize(timezone)
    if celsius:
        aws.iloc[:, 0] = aws.iloc[:, 0] + 273.15
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


## Function to calculate lapse rate from timeseries

def lapseT(high_values, low_values, alt_high, alt_low):
    lapseT = (high_values.mean()-low_values.mean()) / (alt_high-alt_low)
    return lapseT


## Apply preprocessessing on all datasets from SDSS
path = working_directory + 'AWS_atbs/download/'
data_list = []
for file in glob.glob(path + 'atbs*.csv'):
    data_list.append(sdss_open(file, time_slice=True, time_start='2017-06-02T15:00:00'))
aws = round(pd.concat(data_list, axis=1), 2)
aws.columns = ['temp', 'wd', 'prec', 'rh', 'ws']
aws.temp = aws.temp + 273.15
aws.to_csv(working_directory + 'AWS_atbs/atbs_met-data_2017-2020.csv')

## Preprocess data from HOBO Temp/Hum sensors


def hobo_open(path, tz_localize=True, timezone='UTC', time_slice=False, time_start=None, time_end=None):
    hobo = pd.read_csv(path, usecols=(0, 1, 3, 5))
    hobo.columns = ['datetime', 'temp', 'rh', 'dt']
    hobo = hobo[hobo != ' '].dropna()
    hobo.set_index(pd.to_datetime(hobo.datetime), inplace=True)
    hobo = hobo.drop(['datetime'], axis=1)
    hobo = hobo.apply(pd.to_numeric, errors='coerce')
    hobo.iloc[:, [0, 2]] = hobo.iloc[:, [0, 2]] + 273.15
    if tz_localize:
        hobo = hobo.tz_localize(timezone)
    if time_slice and time_start is None and time_end is None:
        print("******************************************************************************************")
        print("WARNING! No time slice defined. Please set valid arguments for time_start and/or time_end.")
        print("******************************************************************************************")
    elif time_slice:
        hobo = hobo[time_start: time_end]
    return hobo


path1 = working_directory + 'HOBO_temphum/HOBO1.csv'
path2 = working_directory + 'HOBO_temphum/HOBO2.csv'
path3 = working_directory + 'HOBO_temphum/HOBO3.csv'
hobo1 = hobo_open(path1)
hobo2 = hobo_open(path2)        # GEHT NICHT!
hobo3 = hobo_open(path3)

## Compare temperature timeseries and calculate lapse rates
compare = pd.DataFrame({'aws': aws.temp, 'minikin_down': minikin_down.temp, 'minikin_up': minikin_up.temp}, index=minikin_up.index)
compare = compare.resample('w').mean()
plt.plot(compare)
plt.legend(compare.columns.tolist(), loc="upper left")
plt.show()
compare.describe()

lapseT(minikin_up.temp, minikin_down.temp, alt_minikin_up, alt_minikin_down) # literature: -0.006 K/m