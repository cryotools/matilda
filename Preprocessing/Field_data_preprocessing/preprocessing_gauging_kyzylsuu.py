from pathlib import Path
import sys
import socket

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/matilda/Preprocessing'
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from Preprocessing_functions import consec_days

plt.ion()

### All datasets:

met = pd.read_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/' +
                  'kyzylsuu_ERA5_Land_1982_2020_42.2_78.2_fitted2AWS.csv', parse_dates=['time'], index_col='time')
met.t2m = met.t2m + 700 * 0.006  # Rough linear scaling to gauging station altitude

hydromet = pd.read_csv(home + '/EBA-CA/Azamat_AvH/workflow/data/Runoff/' +
                       'obs_kyzylsuu_runoff_Hydromet.csv', parse_dates=['Date'], index_col='Date')

bakyt = pd.read_csv(home + '/EBA-CA/Azamat_AvH/workflow/data/Runoff/' +
                    'Kyzylsuu_bakyt_runoff.csv', parse_dates=['time'], index_col='time')

kashkator = pd.read_csv(home + '/EBA-CA/Azamat_AvH/workflow/data/Runoff/' +
                        'kashkator_bakyt_runoff.csv', parse_dates=['Date'], index_col='Date')

# t = slice('2017-01-01', '2018-12-31')
# d = {'HydroM': hydromet[t]['Qobs'], 'Bakyt': bakyt[t]['Qobs'], 'Kashkator': kashkator[t]['Qobs']}
# data = pd.DataFrame(d)
# data.plot(figsize=(12, 6))
# # data.describe()
# plt.show()
# data.sum()

## Hydromet:

# t = slice('1997-12-01', '1998-01-31')
# hydromet[t].plot(figsize=(15, 6))
#
hydromet.plot(figsize=(15, 6))
plt.show()

# weird column in early 1991, extra peak in late 1992 (too low before that as well), 2008 and 2009 completely,
# 2014-2016 completely until 2017-05-04. a lot of jumps around new years eve --> filter winter periods with temperature threshold

# Weird extrema in April 1991, drop in December 1997:
extr = [slice('1991-04-16', '1991-04-16'), slice('1991-04-22', '1991-04-22'), slice('1997-12-10', '1997-12-10')]
for i in extr: hydromet[i] = np.NaN

hydromet[slice('1991-04-01', '1991-04-25')] = hydromet[slice('1991-04-01', '1991-04-25')].interpolate()
hydromet[slice('1997-12-01', '1997-12-20')] = hydromet[slice('1997-12-01', '1997-12-20')].interpolate()

# Exclude corrupted datasets
gaps = [slice('1991-12-01', '1992-12-31'), slice('2008-01-01', '2009-12-31'), slice('2014-01-01', '2017-05-03')]
for i in gaps: hydromet[i] = np.NaN


# obs["Qobs"] = np.where(obs["month"] < 5, 0, obs["Qobs"])
# obs["Qobs"] = np.where(obs["month"] > 10, 0, obs["Qobs"])

hydromet.to_csv(home + '/EBA-CA/Azamat_AvH/workflow/data/Runoff/obs_kyzylsuu_runoff_Hydromet_preprocessed.csv')



## Bakyt:
bakyt.plot(figsize=(15, 6))



## Kashkator:
kashkator.plot(figsize=(15, 6))













# Applying dynamic filter for periods below freezing
# t = slice('2010-01-01', '2019-12-31')
# fig, ax = plt.subplots()
# ax.plot(hydromet[t], c="lightblue")
# ax.set_ylabel("Runoff")
# ax2 = ax.twinx()
# ax2.plot(met[t]['t2m'], c="red", alpha=0.7)
# ax2.set_ylabel("Air Temperature")

# Sets all runoff values 0 where consecutive days are below a threshold temperature

temp_hydro = met.t2m[slice('1989-01-01', '2019-12-31')]

hydromet[consec_days(temp_hydro, 273.15, 5).notna()] = 0

# Is supposed to set the values AFTER the period to 0!!!!!

test = consec_days(temp_hydro, 273.15, 5)

s = temp_hydro
thresh = 273.15
Nmin = 5

m = np.logical_and.reduce([s.shift(-i).le(thresh) for i in range(Nmin)])  # below freezing
n = np.logical_and.reduce([s.shift(-i).ge(thresh) for i in range(Nmin)])  # above freezing
# m = pd.Series(m, index=s.index).replace({False: np.NaN}).ffill(limit=Nmin-1).fillna(False)

for i in range(5, len(m)):
    if m[i - 1] and not n[i]:           # Schon der erste Wert einer 5Tage über 0 Periode zählt. Rückwärts laufen lassen?
        hydromet[i] = 0
    elif m[i - 1] == 0:
        hydromet[i] = 0

# Form consecutive groups
gps = m.ne(m.shift(1)).cumsum().where(m)
