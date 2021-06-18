from pathlib import Path
import sys
import socket
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import plotly.io
import pylab as pl
import scipy.optimize as sp

from Preprocessing_functions import consec_days
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/scripts/Preprocessing'


plt.ion()
plt.ioff()

### All datasets:

met = pd.read_csv(home + '/EBA-CA/Tianshan_data/AWS_atbs/' + 'aws_preprocessed_2017-06_2021-05.csv',
                  parse_dates=['time'], index_col='time')

mukhammed = pd.read_csv(home + '/EBA-CA/Tianshan_data/Gauging_station_Bash-Kaingdy/' +
                        'gauging_report_mukhammed_2019.csv', parse_dates=['time'], index_col='time')
# Toll, das sind gerundete Werte...

ott20 = pd.read_csv(home + '/EBA-CA/Tianshan_data/Gauging_station_Bash-Kaingdy/' + 'ott_pressure_2019-2020.csv',
                  parse_dates=['time'], index_col='time')
ott20 = ott20.resample('D').mean()

ott19 = pd.read_csv(home + '/EBA-CA/Tianshan_data/Gauging_station_Bash-Kaingdy/' + 'OTT2019_recalculated.csv',
                  parse_dates=['time'], index_col='time')
ott19 = ott19[slice('2019-11-22')]

ott = pd.concat([ott19[['pressure']], ott20], axis=0)

manual = pd.read_csv(home + '/EBA-CA/Tianshan_data/Gauging_station_Bash-Kaingdy/' +
                       'runoff_bashkaindy_2017-2019_manual_gauging.csv', parse_dates=['time'], index_col='time')

manual = pd.merge(manual, ott, how='inner', left_index=True, right_index=True)


##  Function of pressure and water level:

# Linear/Polynomial fit:
X = manual['pressure']
Y = manual['water_level']
z = np.polyfit(X, Y, 2)
p = np.poly1d(z)

fig, ax = plt.subplots()
plt.scatter(X, Y)
plt.plot(X, p(X))
plt.show()

# Apply function to the pressure data:
ott['water_level_poly'] = p(ott['pressure'])

fig, ax = plt.subplots()
ott['pressure'].plot(ax=ax, c='red')
ax2 = ax.twinx()
ott['water_level_poly'].plot(ax=ax2)
plt.show()


##  Function of water level and discharge:

# Linear/Polynomial fit:
X = manual['water_level']
Y = manual['discharge']
z = np.polyfit(X, Y,1)
p = np.poly1d(z)

fig, ax = plt.subplots()
plt.scatter(X, Y)
plt.plot(X, p(X), 'r--')
plt.show()

# Apply function to the pressure data:
ott['discharge_from_water_level'] = p(ott['water_level_poly'])

fig, ax = plt.subplots()
ott['discharge_from_water_level'].plot(ax=ax, c='red')
ax2 = ax.twinx()
ott['discharge_poly'].plot(ax=ax2)
plt.show()


## Can we not just make a function of pressure to discharge?
plt.scatter(mukhammed['pressure'], mukhammed['discharge'])
X = manual['pressure']
Y = manual['discharge']
z = np.polyfit(X, Y, 1)
p = np.poly1d(z)

fig, ax = plt.subplots()
plt.scatter(manual['pressure'], manual['discharge'])
plt.plot(X, p(X))
plt.show()

# Apply function to the pressure data:
ott['discharge_poly'] = p(ott['pressure'])

fig, ax = plt.subplots()
ott['pressure'].plot(ax=ax, c='red')
ax2 = ax.twinx()
ott['discharge_poly'].plot(ax=ax2)
plt.show()


ott[['discharge_poly']].to_csv('/home/phillip/Seafile/EBA-CA/Tianshan_data/Gauging_station_Bash-Kaingdy/' +
                               'preprocessed/discharge_bahskaingdy_polyfitted_2019-11_11-2020.csv')











# Applying dynamic filter for periods below freezing
# t = slice('2010-01-01', '2019-12-31')
# fig, ax = plt.subplots()
# ax.plot(hydromet[t], c="lightblue")
# ax.set_ylabel("Runoff")
# ax2 = ax.twinx()
# ax2.plot(met[t]['t2m'], c="red", alpha=0.7)
# ax2.set_ylabel("Air Temperature")

# Sets all runoff values 0 where consecutive days are below a threshold temperature

# temp_hydro = met.t2m[slice('1989-01-01', '2019-12-31')]
#
# hydromet[consec_days(temp_hydro, 273.15, 5).notna()] = 0
#
# # Is supposed to set the values AFTER the period to 0!!!!!
#
# test = consec_days(temp_hydro, 273.15, 5)
#
# s = temp_hydro
# thresh = 273.15
# Nmin = 5
#
# m = np.logical_and.reduce([s.shift(-i).le(thresh) for i in range(Nmin)])  # below freezing
# n = np.logical_and.reduce([s.shift(-i).ge(thresh) for i in range(Nmin)])  # above freezing
# # m = pd.Series(m, index=s.index).replace({False: np.NaN}).ffill(limit=Nmin-1).fillna(False)
#
# for i in range(5, len(m)):
#     if m[i - 1] and not n[i]:           # Schon der erste Wert einer 5Tage über 0 Periode zählt. Rückwärts laufen lassen?
#         hydromet[i] = 0
#     elif m[i - 1] == 0:
#         hydromet[i] = 0
#
# # Form consecutive groups
# gps = m.ne(m.shift(1)).cumsum().where(m)
