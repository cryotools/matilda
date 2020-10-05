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

alt_hobo = 3342         # Water level sensor
lat_hobo = 41.068228
lon_hobo = 75.99092
alt_minikin_up = 3894
alt_minikin_down = 2250
alt_hobo1 = 3036.876
alt_hobo2 = 3376.4644
alt_hobo3 = 3745.4583

## Preprocess Minikin-Combi-Sensors
minikin_up = pd.read_csv(working_directory + 'Minikin/Cognac_glacier/cognac_glacier.csv', sep=';', decimal=',',
                         usecols=range(0, 4))
minikin_up.columns = ['datetime', 'G', 'temp', 'hum']
minikin_up.set_index(pd.to_datetime(minikin_up.datetime), inplace=True)
minikin_up = minikin_up.drop(['datetime'], axis=1)
minikin_up = minikin_up.shift(-2, axis=0)  # Only -2h timeshift results in the correct curves. BUT WHY????
minikin_up = minikin_up.tz_localize('Asia/Bishkek')
minikin_up.temp = minikin_up.temp + 273.15
minikin_up = minikin_up[time_start: time_end]

minikin_down = pd.read_csv(working_directory + 'Minikin/Bash_Kaindy/80_2019_09_12_refin.csv', sep=';', decimal=',',
                           usecols=range(0, 4))
minikin_down.columns = ['datetime', 'G', 'temp', 'hum']
minikin_down.set_index(pd.to_datetime(minikin_down.datetime), inplace=True)
minikin_down = minikin_down.drop(['datetime'], axis=1)
minikin_down = minikin_down.shift(-2, axis=0)  # Only -2h timeshift results in the correct curves. BUT WHY????
minikin_down = minikin_down.tz_localize('Asia/Bishkek')
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

## Apply preprocessing on HOBO timeseries
path1 = working_directory + 'HOBO_temphum/HOBO1.csv'
path2 = working_directory + 'HOBO_temphum/HOBO2.csv'
path3 = working_directory + 'HOBO_temphum/HOBO3.csv'
hobo1 = hobo_open(path1)
hobo2 = hobo_open(path2, time_slice=True, time_end='2018-11-04 20:00:00')                        # "Freezes" at 2018-11-04 20:00:00!
hobo3 = hobo_open(path3, resample=True, time_slice=True, time_end='2019-05-05 07:00:00')        # "Freezes" at 2019-05-05 07:22:05!

## Compare temperature timeseries
compare = pd.DataFrame({'aws [2250m]': aws.temp, 'minikin_down [2250m]': minikin_down.temp,
                        'hobo1 [3037 m]': hobo1.temp, 'hobo2 [3377m]': hobo2.temp, 'hobo3 [3746m]': hobo3.temp,
                        'minikin_up [3864m]': minikin_up.temp}, index=minikin_up.index)
compare = compare.resample('w').mean()
plt.plot(compare)
plt.legend(compare.columns.tolist(), loc="upper left")
plt.show()

compare_slim = pd.DataFrame({'aws [2250m]': aws.temp, 'hobo1 [3037 m]': hobo1.temp,
                        'minikin_up [3864m]': minikin_up.temp}, index=minikin_up.index)
compare_slim = compare_slim.resample('M').mean()
plt.plot(compare_slim)
plt.legend(compare_slim.columns.tolist(), loc="upper left")
plt.show()

## Calculate lapse rates
lapseR(minikin_up.temp, aws.temp, alt_minikin_up,alt_minikin_down)
lapseR(hobo1.temp, aws.temp, alt_hobo1,alt_minikin_down)
lapseR(minikin_up.temp, hobo1.temp, alt_minikin_up,alt_hobo1)

# Seasonal lapse rates
lapseR(minikin_up.temp, aws.temp, alt_minikin_up,alt_minikin_down, seasonal=True, season='winter')
lapseR(hobo1.temp, aws.temp, alt_hobo1,alt_minikin_down, seasonal=True, season='winter')
lapseR(minikin_up.temp, hobo1.temp, alt_minikin_up,alt_hobo1, seasonal=True, season='winter')

lapseR(minikin_up.temp, aws.temp, alt_minikin_up,alt_minikin_down, seasonal=True, season='summer')
lapseR(hobo1.temp, aws.temp, alt_hobo1,alt_minikin_down, seasonal=True, season='summer')
lapseR(minikin_up.temp, hobo1.temp, alt_minikin_up,alt_hobo1, seasonal=True, season='summer')

# As df:
lapse_rates = {'All year': [lapseR(minikin_up.temp, aws.temp, alt_minikin_up,alt_minikin_down),
                            lapseR(hobo1.temp, aws.temp, alt_hobo1,alt_minikin_down),
                            lapseR(minikin_up.temp, hobo1.temp, alt_minikin_up,alt_hobo1)],
               'Summer': [lapseR(minikin_up.temp, aws.temp, alt_minikin_up,alt_minikin_down, seasonal=True, season='summer'),
                          lapseR(hobo1.temp, aws.temp, alt_hobo1, alt_minikin_down, seasonal=True, season='summer'),
                          lapseR(minikin_up.temp, hobo1.temp, alt_minikin_up, alt_hobo1, seasonal=True, season='summer')],
               'Winter': [lapseR(minikin_up.temp, aws.temp, alt_minikin_up,alt_minikin_down, seasonal=True, season='winter'),
                          lapseR(hobo1.temp, aws.temp, alt_hobo1,alt_minikin_down, seasonal=True, season='winter'),
                          lapseR(minikin_up.temp, hobo1.temp, alt_minikin_up,alt_hobo1, seasonal=True, season='winter')]}
lapseR_df = pd.DataFrame(lapse_rates, index = ['2250m to 3894m','2250m to 3036.876m', '3036.876m to 3894m '])