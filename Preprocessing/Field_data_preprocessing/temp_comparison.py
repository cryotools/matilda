import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import glob
home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/Preprocessing/')
from Preprocessing_functions import *
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
era5_gridpoint = 3237.278

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

## Apply preprocessessing on all datasets from SDSS
path = working_directory + 'AWS_atbs/download/'
data_list = []
for file in sorted(glob.glob(path + 'atbs*.csv')):
    data_list.append(sdss_open(file, time_slice=True, time_start='2017-06-02T15:00:00'))
aws = round(pd.concat(data_list, axis=1), 2)
aws.columns = ['temp', 'rh', 'prec', 'ws', 'wd']
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
plt.legend(compare_slim.columns.tolist(), loc="upper left")
# plt.title("Monthly Mean Air Temperature in Bash-Kaindy river valley [°C]")
# plt.savefig('/home/phillip/Seafile/EBA-CA/Workshops/Final_workshop_October2020/Bilder/temp_cognac.png', bbox_inches='tight', dpi=300)
plt.show()

## Comparing the ERA5 data
#era5_down = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182ERA5_Land_2018_2019.csv")
era5_down= pd.read_csv("/home/ana/Seafile/Masterarbeit/Data/no182_ERA5_Land_2000_202011_no182_41_75.9.csv")
era5_down.set_index('TIMESTAMP', inplace=True)
era5_down.index = pd.to_datetime(era5_down.index)
era5_down = era5_down["2018-09-07":"2019-09-13"]

era5_down["temp_2250"] = era5_down["T2"]+(2250-3360)*float(-0.006)
era5_down["temp_3037"] = era5_down["T2"]+(3037-3360)*float(-0.006)
era5_down["temp_3864"] = era5_down["T2"]+(3864-3360)*float(-0.006)
era5_down = era5_down.resample('W').mean()

compare_slim["temp_2250"] = era5_down["temp_2250"]
compare_slim["temp_3037"] = era5_down["temp_3037"]
compare_slim["temp_3864"] = era5_down["temp_3864"]

compare_slim = pd.concat([compare_slim, era5_down], axis=1)

plt.plot(compare_slim)
plt.legend(compare_slim.columns.tolist(), loc="upper left")
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
ax1.plot(compare_slim.index.to_pydatetime(), (compare_slim["aws [2250m]"]), c="#d7191c", label="AWS [2250m]")
ax1.plot(compare_slim.index.to_pydatetime(), compare_slim["temp_2250"], color="#2c7bb6", label="ERA5 [2250m]")
ax2.plot(compare_slim.index.to_pydatetime(), compare_slim["hobo1 [3037 m]"], color="#d7191c", label="hobo1 [3037m]")
ax2.plot(compare_slim.index.to_pydatetime(), (compare_slim["temp_3037"]), c="#2c7bb6", label="ERA5 [3037m]")
ax3.plot(compare_slim.index.to_pydatetime(), compare_slim["minikin_up [3864m]"], c="#d7191c", label="minikin_up [3864m]")
ax3.plot(compare_slim.index.to_pydatetime(), (compare_slim["temp_3864"]), c="#2c7bb6", label="ERA5 [3864m]")
ax1.legend(), ax2.legend(), ax3.legend()
ax1.set_ylabel("[K]", fontsize=9), ax2.set_ylabel("[K]", fontsize=9), ax3.set_ylabel("[K]", fontsize=9)
ax1.set_title("Comparison of downscaled ERA5 data with a lapse rate of -0.006K/m and sensors", fontsize=14)
plt.show()
#plt.savefig("/home/ana/Desktop/Downscaled_temp.png")


## ERA5 + lapse rates
era5 = pd.read_csv("/home/ana/Desktop/compare_slim2.csv")
era5.set_index('datetime.1', inplace=True)
era5.index = pd.to_datetime(era5.index)

lapse_rate_minikin = lapseR(minikin_up.temp, era5["era5"], alt_minikin_up, era5_gridpoint)
lapse_rate_hobo = lapseR(era5["era5"], hobo1.temp, era5_gridpoint, alt_hobo1)

# Seasonal lapse rates
lapseR(minikin_up.temp, era5["era5"], alt_minikin_up, era5_gridpoint, seasonal=True, season='winter')
lapseR(hobo1.temp, era5["era5"], alt_hobo1, era5_gridpoint, seasonal=True, season='winter')

lapseR(minikin_up.temp, era5["era5"], alt_minikin_up, era5_gridpoint, seasonal=True, season='summer')
lapseR(hobo1.temp, era5["era5"], alt_hobo1, era5_gridpoint, seasonal=True, season='summer')

# As df:
lapse_rates = {'All year': [lapseR(minikin_up.temp, era5["era5"], alt_minikin_up, era5_gridpoint),
                            lapseR(hobo1.temp, era5["era5"], alt_hobo1, era5_gridpoint)],
               'Summer': [lapseR(minikin_up.temp, era5["era5"], alt_minikin_up, era5_gridpoint, seasonal=True, season='summer'),
                          lapseR(hobo1.temp, era5["era5"], alt_hobo1, era5_gridpoint, seasonal=True, season='summer')],
               'Winter': [lapseR(minikin_up.temp, era5["era5"], alt_minikin_up, era5_gridpoint, seasonal=True, season='winter'),
                          lapseR(hobo1.temp, era5["era5"], alt_hobo1, era5_gridpoint, seasonal=True, season='winter')]}
lapseR_df = pd.DataFrame(lapse_rates, index = ['3237.278 m and 3894 m','3237.278 m and 3036.876 m'])

era5["era5_fitted_hobo"] = era5["era5"] - (era5_gridpoint-alt_hobo1) * lapse_rate_hobo
era5["era5_fitted_hobo2"] = era5["era5"] - (era5_gridpoint-alt_hobo1) * -0.035
era5["era5_fitted_minikin"] = era5["era5"] + (alt_minikin_up-era5_gridpoint) * lapse_rate_minikin

era5 = era5.resample('w').mean()

fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
ax1.plot(era5.index.to_pydatetime(), (era5["minikin_up_3864"]), c="#d7191c", label="minikin_up [3864m]")
ax1.plot(era5.index.to_pydatetime(), era5["era5"], color="#2c7bb6", label="ERA5 [3237m]")
ax1.plot(era5.index.to_pydatetime(), era5["temp_fitted"], color="#008837", label="Fitted ERA5 (QM)")
ax2.plot(era5.index.to_pydatetime(), era5["Hobo1_3037"], color="#d7191c", label="hobo1 [3037m]")
ax2.plot(era5.index.to_pydatetime(), era5["era5"], color="#2c7bb6", label="ERA5 [3237m]")
ax2.plot(era5.index.to_pydatetime(), (era5["era5_fitted_hobo"]), c="#008837", label="ERA5 fitted [3037m]")
ax3.plot(era5.index.to_pydatetime(), era5["Hobo1_3037"], c="#d7191c", label="hobo1 [3037m]")
ax3.plot(era5.index.to_pydatetime(), era5["era5"], color="#2c7bb6", label="ERA5 [3237m]")
ax3.plot(era5.index.to_pydatetime(), (era5["era5_fitted_hobo2"]), c="#008837", label="ERA5 fitted[3037m]")
ax1.legend(), ax2.legend(), ax3.legend()
ax1.set_ylabel("[K]", fontsize=9), ax2.set_ylabel("[K]", fontsize=9), ax3.set_ylabel("[K]", fontsize=9)
ax1.set_title("ERA5 data [3237 m] downscaled with Quantile Mapping (Minikin data) or lapserates", fontsize=14)
ax2.text(0.01, 0.95, 'Lapse rate ' + str(round(lapse_rate_hobo,2)) +"K/m", transform=ax2.transAxes, fontsize=8, verticalalignment='top')
ax3.text(0.01, 0.95, 'Lapse rate -0.035 K/m', transform=ax3.transAxes, fontsize=8, verticalalignment='top')
plt.tight_layout()
#plt.show()
plt.savefig("/home/ana/Desktop/Fitted_temp.png")
era5_values = era5.describe()
## Cutting the new runoff data
df = pd.read_excel("/home/ana/Seafile/Tianshan_data/HOBO_water/runoff_bashkaindy_11_2019-11_2020.xlsx", parse_dates=[['Date', 'Time']])
#df["Runoff"] = pd.to_numeric(df["Runoff"])
df.set_index('Date_Time', inplace=True)
df.index = pd.to_datetime(df.index)

df_daily = df.resample("D").aggregate({"discharge":"mean"})
df_daily["Qobs"] = df_daily["discharge"]* 86400/46232000*1000

df2 = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/runoff_bashkaindy_2019.csv")

df2["Date_Time"] = pd.to_datetime(df2["Date"])
df2.set_index("Date_Time", inplace=True)

runoff = df2.append(df_daily)
runoff.to_csv("/home/ana/Desktop/runoff_bashkaindy_04_2019-11_2020.csv")

df = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182ERA5_Land_2018_2019.csv", sep=",")
df.columns = ['datetime', 'temp', 'tp']
df.set_index(pd.to_datetime(df.datetime), inplace=True)
df = df[time_start: time_end]

df3 = pd.merge(df, compare_slim, left_index=True, right_index=True)
#df3.to_csv("/home/ana/Desktop/compare_slim.csv")

##

era5 = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182ERA5_Land_lat41.0_lon75.9_alt3839.6_1981-2019.csv", sep=";")
era5.set_index('time', inplace=True)
era5.index = pd.to_datetime(era5.index, utc=True)
#era5.index = era5.index.tz_localize('Asia/Bishkek')
era5 = era5.loc['2000-11-01 01:00:00':'2019-12-31 23:00:00']
era5 = era5.sort_index()

total_precipitation = era5.tp.values #+ height_diff * lapse_rate_total_precipitation))         ### convert from m to mm
total_precipitation[total_precipitation < 0] = era5.tp.values[total_precipitation < 0]
total_precipitation[total_precipitation < 0] = 0
total_precipitation = total_precipitation * 1000
era5["tp"] = total_precipitation

time_start = '2018-10-07 18:00:00'  # longest timeseries of waterlevel sensor
time_end = '2018-10-14 03:00:00'
minikin_up_small = minikin_up[time_start: time_end]

minikin_up = minikin_up.tz_convert('UTC')
minikin_up_small2 = minikin_up[time_start: time_end]

era5_small = era5.loc[time_start: time_end]
era5_small = era5_small.sort_index()

plt.plot(era5_small.index.to_pydatetime(), (minikin_up_small["temp"]), c="#d7191c", label="minikin_up local")
plt.plot(era5_small.index.to_pydatetime(), minikin_up_small2["temp"], color="#008837", label="minikin up UTC")
plt.plot(era5_small.index.to_pydatetime(), era5_small["t2m"], color="#2c7bb6", label="ERA5")
plt.legend()
plt.show()

#era5 = era5.resample("D").agg({"t2m":"mean", "tp":"sum"})
time_start = '2019-01-01'  # longest timeseries of waterlevel sensor
time_end = '2019-12-31'
era_subset = era5.copy()
era_subset = era_subset[time_start:time_end]
era_subset = era_subset.sort_index()

plt.plot(era_subset["t2m"])
plt.show()

