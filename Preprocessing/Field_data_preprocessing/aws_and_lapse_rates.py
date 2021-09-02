import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import glob
home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/matilda/Preprocessing/ERA5_downscaling/')
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/matilda/Preprocessing/')
from Preprocessing_functions import *
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'
working_directory = home + '/Seafile/Tianshan_data/'


## General settings
time_start = '2018-09-07 18:00:00'  # longest timeseries of waterlevel sensor
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
#minikin_up = minikin_up.shift(-2, axis=0)  # Only -2h timeshift results in the correct curves. BUT WHY????
minikin_up = minikin_up.tz_localize('Asia/Bishkek')
minikin_up.temp = minikin_up.temp + 273.15
minikin_up = minikin_up[time_start: time_end]
# minikin_up.to_csv(working_directory+"/Minikin/Cognac_glacier/cognac_glacier_minikin_18_19.csv")

minikin_down = pd.read_csv(working_directory + 'Minikin/Bash_Kaindy/80_2019_09_12_refin.csv', sep=';', decimal=',',
                           usecols=range(0, 4))
minikin_down.columns = ['datetime', 'G', 'temp', 'hum']
minikin_down.set_index(pd.to_datetime(minikin_down.datetime), inplace=True)
minikin_down = minikin_down.drop(['datetime'], axis=1)
minikin_down = minikin_down.shift(-2, axis=0)  # Only -2h timeshift results in the correct curves. BUT WHY????
minikin_down = minikin_down.tz_localize('Asia/Bishkek')
minikin_down.temp = minikin_down.temp + 273.15
minikin_down = minikin_down[time_start: time_end]

## Apply preprocessessing on all datasets from SDSS
path = working_directory + 'AWS_atbs/download/'
data_list = []
for file in sorted(glob.glob(path + 'atbs*.csv')):
    data_list.append(sdss_open(file, time_slice=True, time_start='2017-06-02T15:00:00'))
aws = round(pd.concat(data_list, axis=1), 2)
aws.columns = ['temp1', 'temp', 'rh', 'prec', 'ws', 'wd']
aws.temp = aws.temp + 273.15
#aws.to_csv(working_directory + 'AWS_atbs/atbs_met-data_2017-2020.csv')

## Apply preprocessing on HOBO-temphum timeseries
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
compare_slim = compare_slim.resample('W').mean()
plt.plot(compare_slim)
plt.legend(compare_slim.columns.tolist(), loc="upper center")
# plt.title("Monthly Mean Air Temperature in Bash-Kaindy river valley [°C]")
# plt.savefig('/home/phillip/Seafile/EBA-CA/Workshops/Final_workshop_October2020/Bilder/temp_cognac.png', bbox_inches='tight', dpi=300)
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
