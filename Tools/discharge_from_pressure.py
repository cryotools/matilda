import pandas as pd
from pathlib import Path;

home = str(Path.home())
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'

##
time_start = '2018-09-07 12:00:00'  # longest timeseries of waterlevel
time_end = '2019-09-14 03:00:00'

hobo = pd.read_csv(working_directory + "HOBO_water/AtBashi01_sep19.csv", usecols=[1, 2, 3])
hobo.columns = ['datetime', 'abs_press', 'water_temp']
hobo.datetime = pd.to_datetime(hobo.datetime)
hobo.set_index(hobo.datetime, inplace=True)
hobo = hobo.drop(['datetime'], axis=1)
hobo.water_temp = hobo.water_temp + 273.15
hobo.abs_press = hobo.abs_press * 10  # from kPa to hPa
hobo = hobo[time_start: time_end]

hobo_clim = pd.read_csv(working_directory + "HOBO_water/temp_press_hydrostation_2018-2019.csv")
hobo_clim.datetime = pd.to_datetime(hobo_clim.datetime)
hobo_clim.set_index(hobo_clim.datetime, inplace=True)
hobo_clim = hobo_clim.drop(['datetime'], axis=1)

hobo[['air_temp', 'srf_press']] = hobo_clim
hobo['water_press'] = hobo.abs_press - hobo.srf_press


##
def h(p):  # Water column in meter from hydrostatic pressure in Pa!
    return p / (9.81 * 1000)  # acceleration of gravity: 9.81 m/s², density of water 1000 kg/m³


hobo['water_column'] = h(hobo.water_press * 100)
hobo.describe()