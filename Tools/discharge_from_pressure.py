import pandas as pd
from pathlib import Path;
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

home = str(Path.home())
working_directory = home + '/Seafile/EBA-CA/Tianshan_data/'

##
time_start = '2018-09-07 18:00:00'  # longest timeseries of waterlevel (in UTC+6!)
time_end = '2019-09-14 09:00:00'

hobo = pd.read_csv(working_directory + "HOBO_water/AtBashi01_sep19.csv", usecols=[1, 2, 3])
hobo.columns = ['datetime', 'abs_press', 'water_temp']
hobo.datetime = pd.to_datetime(hobo.datetime)
hobo.set_index(hobo.datetime, inplace=True)
hobo = hobo.tz_localize('UTC')
hobo = hobo.tz_convert('Asia/Bishkek')                  # tz_convert instead of .shift() to preserve all values
hobo = hobo.drop(['datetime'], axis=1)                  # SciView shows UTC!
hobo.water_temp = hobo.water_temp + 273.15
hobo.abs_press = hobo.abs_press * 10  # from kPa to hPa
hobo = hobo[time_start: time_end]


hobo_clim = pd.read_csv(working_directory + "HOBO_water/temp_press_hydrostation_2018-2019.csv")
hobo_clim.datetime = pd.to_datetime(hobo_clim.datetime)
hobo_clim.set_index(hobo_clim.datetime, inplace=True)
hobo_clim = hobo_clim.tz_localize('Asia/Bishkek')
hobo_clim = hobo_clim.drop(['datetime'], axis=1)

hobo[['air_temp', 'srf_press']] = hobo_clim
hobo['water_press'] = hobo.abs_press - hobo.srf_press

##
def h(p):  # Water column in meter from hydrostatic pressure in Pa!
    return p / (9.81 * 1000)  # acceleration of gravity: 9.81 m/s², density of water 1000 kg/m³


hobo['water_column'] = h(hobo.water_press * 100)
hobo.describe()

##
# Wie umgehen mit negativen Werten?
# Wo sind die Tracermessungen von 2018?

tracer = pd.read_csv(working_directory + "HOBO_water/dilution_gauging_2019_09_14.csv")
tracer.datetime = pd.to_datetime(tracer.datetime)
tracer.set_index(tracer.datetime, inplace=True)
tracer = tracer.tz_localize('Asia/Bishkek')
tracer = tracer.drop(['datetime'], axis=1)

hobo_ro = pd.merge(tracer, hobo, how='left', left_index=True, right_index=True)         # Dilution gauging was done after reading out the sensor...

##

predictor = 'water_column'

X = hobo.loc[:, predictor].values.reshape(-1, 1)
Y = data_wide.loc[:, 'ufp'].values.reshape(-1, 1)

linear_regressor = LinearRegression()

linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel(str(''))
plt.ylabel('')
plt.show()

print('------ Lineare Regression -----')
print('Funktion: y = %.3f * x + %.3f' % (linear_regressor.coef_[0], linear_regressor.intercept_))
print("R² Score: {:.2f}".format(linear_regressor.score(X, Y)))
print("\n")