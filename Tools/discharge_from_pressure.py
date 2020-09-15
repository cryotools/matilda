import pandas as pd
from pathlib import Path;

home = str(Path.home())
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'

hobo = pd.read_csv(working_directory + "HOBO_water/AtBashi01_sep19.csv", usecols=[1, 2, 3])
hobo.columns = ['datetime', 'abs_press', 'temp']
hobo.datetime = pd.to_datetime(hobo.datetime)
hobo.set_index(hobo.datetime, inplace=True)
hobo = hobo.drop(['datetime'], axis=1)

aws_up = pd.read_csv(working_directory + 'Minikin/Cognac_glacier/80_2019_09_12_refin.csv', sep=';', decimal=',', usecols=range(0, 4))
aws_up.columns = ['datetime', 'G', 'temp', 'hum']
aws_up.datetime = pd.to_datetime(aws_up.datetime)
aws_up.set_index(aws_up.datetime, inplace=True)
aws_up = aws_up.drop(['datetime'], axis=1)

aws_far = pd.read_csv(working_directory + 'AWS_asai/press_hPa_2017-08-01_2020-09-15.csv')
aws_far.columns = ['datetime', 'air_press']
aws_far.datetime = pd.to_datetime(aws_far.datetime)
aws_far.set_index(aws_far.datetime, inplace=True)
aws_far = aws_far.drop(['datetime'], axis=1)

alt_hobo = 3330
alt_aws_far = 3023

def p(p0,h,t0):
    return p0 * (1 - (0.0065 * h)/t0) ** 5.255       # Calculate air pressure at altitude.


