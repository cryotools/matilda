import warnings
warnings.filterwarnings("ignore")#sklearn
import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from skdownscale.pointwise_models import BcsdTemperature

os.chdir(sys.path[0])

obs = pd.read_csv('obs.csv', index_col='time', parse_dates=['time'])
reanalysis = pd.read_csv('reanalysis.csv', index_col='time', parse_dates=['time'])
scenario = pd.read_csv('scenario.csv', index_col='time', parse_dates=['time'])


## Step 1 - Downscale reananlysis using obs

train_slice = slice('2007-08-10', '2016-01-01')
predict_slice = slice('1982-01-01', '2020-12-31')

x_train = reanalysis[train_slice]
y_train = obs[train_slice]
x_predict = reanalysis[predict_slice]

fit = BcsdTemperature(return_anoms=False).fit(x_train, y_train)
t_corr = pd.DataFrame(index=x_predict.index)
t_corr['t2m'] = fit.predict(x_predict)


## Step 2 - Downscale scenario data using fitted reanalysis data

train_slice = slice('1982-01-01', '2020-12-31')
predict_slice = slice('2000-01-01', '2100-12-31')

x_train = scenario[train_slice]
y_train = t_corr[train_slice]
x_predict = scenario[predict_slice]

fit = BcsdTemperature(return_anoms=False).fit(x_train, y_train)
t_corr_scenario = pd.DataFrame(index=x_predict.index)
t_corr_scenario['t2m'] = fit.predict(x_predict)


## Step 3 - Check for monthly 'heartbeats'

def daily_annual_T(x, t):
    x = x[t][['t2m']]
    x["month"] = x.index.month
    x["day"] = x.index.day
    day1 = x.index[0]
    x = x.groupby(["month", "day"]).mean()
    date = pd.date_range(day1, freq='D', periods=len(x)).strftime('%Y-%m-%d')
    x = x.set_index(pd.to_datetime(date))
    return x


t = slice('2000-01-01', '2020-12-31')
reanalysis_annual = daily_annual_T(reanalysis, t)
t_corr_annual = daily_annual_T(t_corr, t)
fig, ax = plt.subplots()
reanalysis_annual['t2m'].plot(label='reanalysis', ax=ax, legend=True)
t_corr_annual['t2m'].plot(label='reanalysis_fitted', ax=ax, legend=True, c='red')
ax.set_title('Multi-annual daily means of air temperature (2000-2020)')
plt.show()

t = slice('2000-01-01', '2020-12-31')
t_corr_annual = daily_annual_T(t_corr_scenario, t)
scenario_annual = daily_annual_T(scenario, t)
fig, ax = plt.subplots()
scenario_annual['t2m'].plot(label='scenario', ax=ax, legend=True)
t_corr_annual['t2m'].plot(label='scenario_fitted', ax=ax, legend=True, c='red')
ax.set_title('Multi-annual daily means of air temperature (2000-2020)')
plt.show()

