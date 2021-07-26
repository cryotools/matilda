##
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import salem
from pathlib import Path
import sys
import socket
import os
warnings.filterwarnings("ignore")  # sklearn
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/matilda/Preprocessing'
os.chdir(wd + '/Downscaling')
sys.path.append(wd)
import Downscaling.scikit_downscale_matilda as sds
from Preprocessing_functions import pce_correct
from forcing_data_preprocessing_bashkaingdy import era, era_D, aws_D_int, aws, cmip


#################################
#    Downscaling temperature    #
#################################


## Step 1 - Downscale ERA5 using AWS

# Test for most suitable downscaling algorithm:

train_slice = slice('2017-07-14', '2019-12-31')         # For best algorithm.
predict_slice = slice('2020-01-01', '2020-12-31')       # For best algorithm.
final_train_slice = slice('2017-07-14', '2020-12-31')
final_predict_slice = slice('1982-01-01', '2020-12-31')
plot_slice = slice('2017-07-14', '2020-12-31')

# sds.overview_plot(era[plot_slice]['t2m'], aws[plot_slice]['t2m'],
#                   labelvar1='Temperature [K]')

x_train = era_D[train_slice].drop(columns=['tp'])           # Some algorithms can't cope with NA. Daily data here.
y_train = aws_D_int[train_slice].drop(columns=['tp', 'ws'])
x_predict = era_D[predict_slice].drop(columns=['tp'])
y_predict = aws_D_int[predict_slice].drop(columns=['tp', 'ws'])

prediction = sds.fit_dmodels(x_train, y_train, x_predict)
# sds.modcomp_plot(aws_D_int[predict_slice]['t2m'], x_predict[predict_slice]['t2m'], prediction['predictions'][predict_slice], ylabel='Temperature [K]')
# sds.dmod_score(prediction['predictions'], aws_D_int['t2m'], y_predict['t2m'], x_predict['t2m'])


# Apply best model on full training and prediction periods

x_train = era_D[final_train_slice].drop(columns=['tp'])
y_train = aws_D_int[final_train_slice].drop(columns=['tp', 'ws'])
x_predict = era_D[final_predict_slice].drop(columns=['tp'])
y_predict = aws_D_int[final_predict_slice].drop(columns=['tp', 'ws'])

best_mod = prediction['models']['GARD: LinearRegression']          # GARD: LinearRegression, BCSD: BcsdTemperature, GARD: PureAnalog-sample-10
best_mod.fit(x_train, y_train)                                     # Linear Regression works only with daily_interpol
t_corr = pd.DataFrame(index=x_predict.index)
t_corr['t2m'] = best_mod.predict(x_predict)
t_corr_D = t_corr.resample('D').mean()


# # Compare results with training and target data:
# freq = 'W'
# fig, ax = plt.subplots(figsize=(12,8))
# t_corr['t2m'][final_train_slice].resample(freq).mean().plot(ax=ax, label='fitted', legend=True)
# x_predict['t2m'][final_train_slice].resample(freq).mean().plot(label='era5', ax=ax, legend=True)
# y_predict['t2m'].resample(freq).mean().plot(label='aws', ax=ax, legend=True)
#
# compare = pd.concat({'fitted':t_corr['t2m'][final_train_slice], 'era5': x_predict['t2m'][final_train_slice],
#                      'aws':y_predict['t2m'][final_train_slice]}, axis=1)
# compare.describe()



## Step 2 - Downscale CMIP6 using fitted ERA5

# Test for most suitable downscaling algorithm:

train_slice = slice('2000-01-01', '2009-12-31')         # For best algorithm.
predict_slice = slice('2010-01-01', '2019-12-31')       # For best algorithm.
final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-30')
plot_slice = slice('2010-01-01', '2019-12-31')

# sds.overview_plot(cmip[plot_slice]['t2m'], t_corr_D[plot_slice]['t2m'],
#                   labelvar1='Temperature [K]')

x_train = cmip[train_slice].drop(columns=['tp'])
y_train = t_corr_D[train_slice]
x_predict = cmip[predict_slice].drop(columns=['tp'])
y_predict = t_corr_D[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict)
# sds.modcomp_plot(t_corr_D[plot_slice]['t2m'], x_predict[plot_slice]['t2m'], prediction['predictions'][plot_slice], ylabel='Temperature [K]')
# sds.dmod_score(prediction['predictions'], t_corr_D['t2m'], y_predict['t2m'], x_predict['t2m'])


# Apply best model on full training and prediction periods

x_train = cmip[final_train_slice].drop(columns=['tp'])
y_train = t_corr_D[final_train_slice]
x_predict = cmip[final_predict_slice].drop(columns=['tp'])
y_predict = t_corr_D[final_predict_slice]

best_mod = prediction['models']['GARD: LinearRegression']        #  'BCSD: BcsdTemperature'
best_mod.fit(x_train, y_train)
t_corr_cmip = pd.DataFrame(index=x_predict.index)
t_corr_cmip['t2m'] = best_mod.predict(x_predict)


# Compare results with training and target data:
# freq = 'M'
# fig, ax = plt.subplots(figsize=(12,8))
# t_corr_cmip['t2m'].resample(freq).mean().plot(ax=ax, label='cmip6_fitted', legend=True)
# x_predict['t2m'].resample(freq).mean().plot(label='cmip6', ax=ax, legend=True)
# y_predict['t2m'].resample(freq).mean().plot(label='era5l_fitted', ax=ax, legend=True)
# plt.show()
# compare = pd.concat({'cmip6fitted': t_corr_cmip['t2m'][final_train_slice], 'cmip6': x_predict['t2m'][final_train_slice],
#                      'era5fitted': y_predict['t2m'][final_train_slice]}, axis=1)
# compare.describe()



# t = slice('2040-01-01','2041-12-31')
# freq = 'D'
# fig, ax = plt.subplots(figsize=(12,8))
# t_corr_cmip['t2m'][t].resample(freq).mean().plot(ax=ax, label='cmip6_fitted', legend=True)
# x_predict['t2m'][t].resample(freq).mean().plot(label='cmip6', ax=ax, legend=True)
# plt.show()




#################################
#   Downscaling precipitation   #
#################################

## Step 1 - Downscale ERA5 using AWS

# Test for most suitable downscaling algorithm:

train_slice = slice('2017-06-03', '2018-12-31')         # 2017-06-03 first full day
predict_slice = slice('2019-01-01', '2019-12-31')       # Sensor appears to be corrupted in 2020
final_train_slice = slice('2017-06-03', '2019-12-31')
final_predict_slice = slice('1982-01-01', '2020-12-31')
plot_slice = slice('2017-06-02', '2019-12-31')

# sds.overview_plot(era[plot_slice]['tp'], aws[plot_slice]['tp'],
#                   labelvar1='Daily Precipitation [mm]', figsize=(15, 6))

x_train = era[train_slice].drop(columns=['t2m'])
y_train = aws[train_slice].drop(columns=['t2m', 'ws'])
x_predict = era[predict_slice].drop(columns=['t2m'])
y_predict = aws[predict_slice].drop(columns=['t2m', 'ws'])

prediction = sds.fit_dmodels(x_train, y_train, x_predict, precip=True,
                             **{'detrend': False})
        # Doesn't work with AnalogRegression. Detrending results in negative values.
# sds.modcomp_plot(aws[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice],
#                  ylabel='Daily Precipitation [mm]')
# sds.dmod_score(prediction['predictions'], aws['tp'], y_predict['tp'], x_predict['tp'])


# Compare results with training and target data:
# compare = pd.concat([prediction['predictions'], y_predict['tp']], axis=1)
# comp_desc = compare.describe()
# compare.sum()                           # PRECIPITATION GETS A LOT MORE IN EVERY CASE!


# Apply best model on full training and prediction periods

x_train = era[final_train_slice].drop(columns=['t2m'])
y_train = aws[final_train_slice].drop(columns=['t2m', 'ws'])
x_predict = era[final_predict_slice].drop(columns=['t2m'])
y_predict = aws[final_predict_slice].drop(columns=['t2m', 'ws'])

# sds.overview_plot(era[final_train_slice]['tp'], aws[final_train_slice]['tp'],
#                   labelvar1='Daily Precipitation [mm]')

best_mod = prediction['models']['BCSD: BcsdPrecipitation']          # BCSD: BcsdPrecipitation, GARD: PureAnalog-best-1, GARD: PureAnalog-sample-10 Pick the best model by name.
best_mod.fit(x_train, y_train)
p_corr = pd.DataFrame(index=x_predict.index)
p_corr['tp'] = best_mod.predict(x_predict)
p_corr_D = p_corr.resample('D').sum()

# Compare results with training and target data:
# freq = 'D'
# fig, ax = plt.subplots(figsize=(12,8))
# x_predict['tp'][final_train_slice].resample(freq).sum().plot(label='era5', ax=ax, legend=True)
# y_predict['tp'].resample(freq).sum().plot(label='aws', ax=ax, legend=True)
# p_corr['tp'][final_train_slice].resample(freq).sum().plot(ax=ax, label='fitted', legend=True)
# plt.show()

# compare = pd.concat({'fitted':p_corr['tp'][final_train_slice], 'era5': x_predict['tp'][final_train_slice],
#                      'aws':y_predict['tp'][final_train_slice]}, axis=1)
# compare.describe()
# compare.sum()                       # Here the precipitation gets less than the observed...

# KÖNNTE BESSER AUSSEHEN....HAR-DATEN VERWENDEN?



## Step 2 - Downscale CMIP6 using fitted ERA5

# Test for most suitable downscaling algorithm:

train_slice = slice('2000-01-01', '2009-12-31')         # For best algorithm.
predict_slice = slice('2010-01-01', '2019-12-31')       # For best algorithm.
final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-30')
plot_slice = slice('2010-01-01', '2019-12-31')

# sds.overview_plot(cmip[plot_slice]['tp'], p_corr_D[plot_slice]['tp'],
#                   labelvar1='Daily Precipitation [mm]')

x_train = cmip[train_slice].drop(columns=['t2m'])
y_train = p_corr_D[train_slice]
x_predict = cmip[predict_slice].drop(columns=['t2m'])
y_predict = p_corr_D[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict, precip=True)
# sds.modcomp_plot(p_corr_D[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice], ylabel='Daily Precipitation [mm]')
# sds.dmod_score(prediction['predictions'], p_corr_D['tp'], y_predict['tp'], x_predict['tp'])

# # Compare results with training and target data:
# compare = pd.concat([prediction['predictions'], y_predict['tp']], axis=1)
# comp_desc = compare.describe()
# compare.sum()


# Apply best model on full training and prediction periods

x_train = cmip[final_train_slice].drop(columns=['t2m'])
y_train = p_corr_D[final_train_slice]
x_predict = cmip[final_predict_slice].drop(columns=['t2m'])
y_predict = p_corr_D[final_predict_slice]

best_mod = prediction['models']['BCSD: BcsdPrecipitation']        # BCSD: BcsdPrecipitation, GARD: PureAnalog-best-1, GARD: PureAnalog-sample-10 Pick the best model by name.
best_mod.fit(x_train, y_train)
p_corr_cmip = pd.DataFrame(index=x_predict.index)
p_corr_cmip['tp'] = best_mod.predict(x_predict)


# # Compare results with training and target data:
# freq = 'Y'
# fig, ax = plt.subplots(figsize=(12, 8))
# p_corr_cmip['tp'].resample(freq).sum().plot(ax=ax, label='cmip6_fitted', legend=True)
# x_predict['tp'].resample(freq).sum().plot(label='cmip6', ax=ax, legend=True)
# y_predict['tp'].resample(freq).sum().plot(label='era5l_fitted', ax=ax, legend=True)
#
# compare = pd.concat({'cmip6fitted':p_corr_cmip['tp'][final_train_slice], 'cmip6': x_predict['tp'][final_train_slice],
#                      'era5fitted':y_predict['tp'][final_train_slice]}, axis=1)
# compare.describe()
# compare.sum()



################################
#   Saving final time series   #
################################

# cmip_corr = pd.concat([t_corr_cmip, p_corr_cmip], axis=1)
# era_corr = pd.concat([t_corr, p_corr_D], axis=1)
# cmip_corr.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/CMIP6/'
#                  + 'no182_CMIP6_ssp2_4_5_mean_2000_2100_41-75.9_fitted2ERA5Lfit.csv')
# era_corr.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/' +
#                 'no182_ERA5_Land_1982_2020_41_75.9_fitted2AWS.csv')


## Check weird peaks in downscaled data:

# def daily_annual_T(x, t):
#     x = x[t][['t2m']]
#     x["month"] = x.index.month
#     x["day"] = x.index.day
#     day1 = x.index[0]
#     x = x.groupby(["month", "day"]).mean()
#     date = pd.date_range(day1, freq='D', periods=len(x)).strftime('%Y-%m-%d')
#     x = x.set_index(pd.to_datetime(date))
#     return x
#
# t = slice('2000-01-01', '2100-12-30')
# cmip_annual = daily_annual_T(cmip, t)
# t_corr_annual = daily_annual_T(t_corr_cmip, t)
# fig, ax = plt.subplots()
# cmip_annual['t2m'].plot(label='original', ax=ax, legend=True)
# t_corr_annual['t2m'].plot(label='fitted', ax=ax, legend=True, c='red')
# plt.show()
#
# t = slice('2000-01-01', '2020-12-30')
# era_annual = daily_annual_T(era, t)
# t_corr_annual = daily_annual_T(t_corr, t)
# fig, ax = plt.subplots()
# era_annual['t2m'].plot(label='original', ax=ax, legend=True)
# t_corr_annual['t2m'].plot(label='fitted', ax=ax, legend=True, c='red')
# plt.show()