import pandas as pd
import numpy as np
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
hobo = hobo.tz_convert('Asia/Bishkek')  # tz_convert instead of .shift() to preserve all values
hobo = hobo.drop(['datetime'], axis=1)  # SciView shows UTC!
hobo.water_temp = hobo.water_temp + 273.15
hobo.abs_press = hobo.abs_press * 10  # from kPa to hPa
hobo = hobo[time_start: time_end]
plt.plot(hobo.abs_press["2019-08-01": "2019-08-03"]); plt.show()


hobo_clim = pd.read_csv(working_directory + "HOBO_water/temp_press_hydrostation_2018-2019.csv")
hobo_clim.datetime = pd.to_datetime(hobo_clim.datetime)
hobo_clim.set_index(hobo_clim.datetime, inplace=True)
hobo_clim = hobo_clim.tz_localize('UTC')
hobo_clim = hobo_clim.tz_convert('Asia/Bishkek')
hobo_clim = hobo_clim.drop(['datetime'], axis=1)
plt.plot(hobo_clim.temp["2019-08-01": "2019-08-03"]); plt.show()

hobo[['air_temp', 'srf_press']] = hobo_clim
hobo['water_press'] = hobo.abs_press - hobo.srf_press

plt.plot(hobo.air_temp["2019-08-01": "2019-08-03"])
plt.show()


##
def h(p):  # Water column in meter from hydrostatic pressure in Pa!
    return p / (9.81 * 1000)  # acceleration of gravity: 9.81 m/s², density of water 1000 kg/m³


hobo['water_column'] = h(hobo.water_press * 100)

hobo.describe()

##
# Wie umgehen mit negativen Werten?

tracer19 = pd.read_csv(working_directory + "HOBO_water/dilution_gauging_2019_09_14.csv")
tracer19.datetime = pd.to_datetime(tracer19.datetime)
tracer19.set_index(tracer19.datetime, inplace=True)
tracer19 = tracer19.tz_localize('Asia/Bishkek')
tracer19 = tracer19.drop(['datetime'], axis=1)

tracer18 = pd.read_csv(working_directory + "HOBO_water/manual_gauging_2018_09.csv")
tracer18.datetime = pd.to_datetime(tracer18.datetime)
tracer18.set_index(tracer18.datetime, inplace=True)
tracer18 = tracer18.tz_localize('Asia/Bishkek')
tracer18 = tracer18.drop(['datetime'], axis=1)
tracer18 = tracer18.resample('H').mean()  # Fit gauging data to HOURLY water level.
tracer18 = tracer18.dropna()

hobo_ro_18 = pd.merge(tracer18, hobo, how='left', left_index=True, right_index=True)
# hobo_ro_19 = pd.merge(tracer19, hobo, how='left', left_index=True, right_index=True)         # Dilution gauging was done after reading out the sensor...

##
predictor = 'water_column'
X = hobo_ro_18.loc[:, predictor].values.reshape(-1, 1)
Y = hobo_ro_18.loc[:, 'discharge'].values.reshape(-1, 1)
linear_regressor = LinearRegression()
linear_regressor.fit(X, Y)
Y_pred = linear_regressor.predict(X)

plt.scatter(X, Y)
plt.plot(X, Y_pred, color='red')
plt.xlabel('Water column above sensor [m]')
plt.ylabel('Run-off [m³/s')
plt.show()

print('------ Linear Regression -----')
print('Function: y = %.3f * x + %.3f' % (linear_regressor.coef_[0], linear_regressor.intercept_))
print("R² Score: {:.2f}".format(linear_regressor.score(X, Y)))

##
runoff_18 = pd.DataFrame({'discharge': linear_regressor.coef_[0] * hobo.water_column + linear_regressor.intercept_})
runoff_18[runoff_18.index.month.isin([10, 11, 12, 1, 2, 3, 4])] = 0
runoff_18[runoff_18.discharge < 0] = 0
runoff_18.describe()

plt.plot(runoff_18["2019-08-01": "2019-08-02"])
plt.show()

plt.plot(hobo.air_temp["2019-08-01": "2019-08-03"])
plt.show()

runoff_18.to_csv(working_directory + "HOBO_water/runoff_cognac_glacier_09-2018_09-2019.csv")