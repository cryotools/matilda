import pandas as pd
from pathlib import Path;

home = str(Path.home())
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'

##
time_start = '2018-09-07 12:00:00'      # longest timeseries of waterlevel
time_end = '2019-09-14 03:00:00'

aws_up = pd.read_csv(working_directory + 'Minikin/Cognac_glacier/80_2019_09_12_refin.csv', sep=';', decimal=',', usecols=range(0, 4))
aws_up.columns = ['datetime', 'G', 'temp', 'hum']
aws_up.datetime = pd.to_datetime(aws_up.datetime)
aws_up.set_index(aws_up.datetime, inplace=True)
aws_up = aws_up.drop(['datetime'], axis=1)
aws_up.temp = aws_up.temp + 273.15
aws_up = aws_up[time_start: time_end]


aws_down = pd.read_csv(working_directory + 'AWS_atbs/temp_2017-05-30_2020-09-16.csv')
aws_down.columns = ['datetime', 'temp']
aws_down.datetime = pd.to_datetime(aws_down.datetime)
aws_down.set_index(aws_down.datetime, inplace=True)
aws_down = aws_down.drop(['datetime'], axis=1)
aws_down.temp = aws_down.temp + 273.15
aws_down = aws_down.resample('H').mean()
aws_down = aws_down[time_start: time_end]

##
aws_far = pd.read_csv(working_directory + 'AWS_asai/press_hPa_2017-08-01_2020-09-15_cut.csv')
aws_far.columns = ['datetime', 'air_press']
aws_far.datetime = pd.to_datetime(aws_far.datetime)
aws_far.set_index(aws_far.datetime, inplace=True)
aws_far = aws_far.drop(['datetime'], axis=1)
aws_far = aws_far.resample('H').mean()

aws_far_temp = pd.read_csv(working_directory + 'AWS_asai/temp_2017-08-01_2020-09-15_cut.csv')
aws_far_temp.columns = ['datetime', 'temp']
aws_far_temp.datetime = pd.to_datetime(aws_far_temp.datetime)
aws_far_temp.set_index(aws_far_temp.datetime, inplace=True)
aws_far_temp = aws_far_temp.drop(['datetime'], axis=1)
aws_far_temp.temp = aws_far_temp.temp + 273.15
aws_far_temp = aws_far_temp.resample('H').mean()

aws_far['temp'] = aws_far_temp.temp
aws_far = aws_far[time_start: time_end]

##
alt_hobo = 3342
alt_aws_far = 3023
alt_aws_up = 3894
alt_aws_down = 2250
lapseT = -(aws_down.temp.mean() - aws_up.temp.mean()) / (alt_aws_down - alt_aws_up) # literature: -0.006 K/m

##
temp_hobo = aws_down.temp + (alt_hobo - alt_aws_down) * lapseT       # Calculate temperature at water level sensor location from aws_down series.
temp_mean = (temp_hobo + aws_far.temp) / 2

def p(p0,h,t0, lapseR):
    return p0 * (1 - (lapseR * h)/t0) ** 5.255       # Calculate air pressure at altitude.


p_hobo = p(aws_far.air_press, alt_hobo - alt_aws_far, temp_mean, (aws_far.temp.mean() - aws_up.temp.mean()) / (alt_aws_far - alt_aws_up))
p_hobo.describe()

##
data_hobo = round(pd.DataFrame({'temp': temp_hobo, 'press': p_hobo}), 2)
data_hobo.to_csv(working_directory + "HOBO_water/temp_press_hydrostation_2018-2019.csv")