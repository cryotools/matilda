import numpy as np
import pandas as pd


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

def lapseR(high_values, low_values, alt_high, alt_low, unit='K/m',
           seasonal=False, season=None, summer=[4,5,6,7,8,9,10,11], winter=[12,1,2,3]):
    if seasonal and season is 'summer':
        lapseR = (high_values[high_values.index.month.isin(summer)].mean()
                  - low_values[low_values.index.month.isin(summer)].mean()) / (alt_high - alt_low)
        print('The lapse rate between', alt_low, 'm and', alt_high, 'm in', season, 'is', round(lapseR, 5), unit)
    elif seasonal and season is 'winter':
        lapseR = (high_values[high_values.index.month.isin(winter)].mean()
                  - low_values[low_values.index.month.isin(winter)].mean()) / (alt_high - alt_low)
        print('Lapse rate between', alt_low, 'm and', alt_high, 'm in', season, 'is', round(lapseR, 5), unit)
    else:
        lapseR = (high_values.mean()-low_values.mean()) / (alt_high-alt_low)
        print('The lapse rate', alt_low, 'm and', alt_high, 'm is', round(lapseR, 5), unit)
    return lapseR


## Preprocess data from HOBO Temp/Hum sensors


def hobo_open(path, tz_localize=True, timezone='UTC', time_slice=False, time_start=None, time_end=None,
              resample=False, resample_rate='H', resample_method='mean'):
    hobo = pd.read_csv(path, usecols=(["Date Time - UTC", "Temp, (*C)", "RH, (%)", "DewPt, (*C)"]))
    hobo.columns = ['datetime', 'temp', 'rh', 'dt']
    if hobo.temp.dtype is not np.dtype('float64'):
        hobo = hobo[hobo != ' '].dropna()
    hobo.set_index(pd.to_datetime(hobo.datetime), inplace=True)
    hobo = hobo.drop(['datetime'], axis=1)
    hobo = hobo.apply(pd.to_numeric, errors='coerce')
    hobo.iloc[:, [0, 2]] = hobo.iloc[:, [0, 2]] + 273.15
    if tz_localize:
        hobo = hobo.tz_localize(timezone)
    if resample and resample_method == 'mean':
        hobo = hobo.resample(resample_rate).mean()
    elif resample and resample_method == 'sum':
        hobo = hobo.resample(resample_rate).sum()
    if time_slice and time_start is None and time_end is None:
        print("******************************************************************************************")
        print("WARNING! No time slice defined. Please set valid arguments for time_start and/or time_end.")
        print("******************************************************************************************")
    elif time_slice:
        hobo = hobo[time_start: time_end]
    return hobo
