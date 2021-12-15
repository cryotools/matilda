##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import scipy.stats as stats
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
import os
os.chdir(wd + '/Downscaling')
sys.path.append(wd)
import Downscaling.scikit_downscale_matilda as sds
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from skdownscale.pointwise_models import PureAnalog, AnalogRegression
from skdownscale.pointwise_models import BcsdTemperature, BcsdPrecipitation
from Preprocessing_functions import pce_correct, trendline, daily_annual_T
from forcing_data_preprocessing_kyzylsuu import era_temp_D_int, aws_temp_D_int, era_temp_D, aws_temp_D, era,\
    aws_temp, cmip, aws_prec, era_D

# plt.ion()         # interactive plotting?



#################################
#    Downscaling temperature    #
#################################

## Step 1 - Downscale ERA5 using AWS


final_train_slice = slice('2007-08-10', '2016-01-01')
final_predict_slice = slice('1982-01-01', '2020-12-31')
plot_slice = slice('2007-08-10', '2016-01-01')

# sds.overview_plot(era[plot_slice]['t2m'], aws_temp[plot_slice]['t2m'],
#                   labelvar1='Temperature [K]')

x_train = era_temp_D_int[final_train_slice]         # Use datasets with gap excluded for training (avoid NAs)
y_train = aws_temp_D_int[final_train_slice]
x_predict = era_temp_D[final_predict_slice]         # Use full dataset for prediction
y_predict = aws_temp_D[final_predict_slice]

fit = BcsdTemperature(return_anoms=False).fit(x_train, y_train)      #  RandomForestRegressor(random_state=0) LinearRegression()
t_corr = pd.DataFrame(index=x_predict.index)
t_corr['t2m'] = fit.predict(x_predict)
t_corr_D = t_corr.resample('D').mean()


# x_train = era[final_train_slice].drop(columns=['tp'])
# y_train = aws_temp[final_train_slice]
# x_predict = era[final_predict_slice].drop(columns=['tp'])
# y_predict = aws_temp[final_predict_slice]
#
# fit = BcsdTemperature(return_anoms=False).fit(x_train, y_train)
# t_corr = pd.DataFrame(index=x_predict.index)
# t_corr['t2m'] = fit.predict(x_predict)
# t_corr_D = t_corr.resample('D').mean()

# freq = 'W'
# fig, ax = plt.subplots(figsize=(12,8))
# t_corr['t2m'][plot_slice].resample(freq).mean().plot(ax=ax, label='fitted', legend=True)
# era_temp_D_int['t2m'][plot_slice].resample(freq).mean().plot(label='era5', ax=ax, legend=True)
# aws_temp_D_int['t2m'][plot_slice].resample(freq).mean().plot(label='aws', ax=ax, legend=True)
#
# plt.show()




##
from bias_correction import BiasCorrection

final_train_slice = slice('2007-08-10', '2016-01-01')
final_predict_slice = slice('1982-01-01', '2020-12-31')
x_train = era_temp_D[final_train_slice]['t2m'].squeeze()
y_train = aws_temp_D[final_train_slice]['t2m'].squeeze()
x_predict = era_temp_D[final_predict_slice]['t2m'].squeeze()

bc = BiasCorrection(y_train, x_train, x_predict)
t_corr_bc = pd.DataFrame(bc.correct(method='normal_correction'))  # normal_correction refers to QM but SDM can't cope with NAs


final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-31')

x_train = cmip[final_train_slice]['t2m'].squeeze()
y_train = t_corr_bc[final_train_slice]['t2m'].squeeze()
x_predict = cmip[final_predict_slice]['t2m'].squeeze()

bc_cmip = BiasCorrection(y_train, x_train, x_predict)
t_corr_bc_cmip = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))

t_corr_bc_cmip.describe()
x_train.describe()
y_train.describe()
x_predict.describe()


slice('2007-08-10', '2016-01-01')
slice('1982-01-01', '2020-12-31')

t = slice('2007-08-10', '2016-01-01')
freq = 'M'
fig, ax = plt.subplots(figsize=(12, 8))
aws_temp_D[t]['t2m'].resample(freq).mean().plot(ax=ax, label='aws', legend=True)
era_temp_D[t]['t2m'].resample(freq).mean().plot(ax=ax, label='era', legend=True)
t_corr_bc[t]['t2m'].resample(freq).mean().plot(label='era_sdm', ax=ax, legend=True)

plt.show()



# final_train_slice = slice('1982-01-01', '2020-12-31')
# final_predict_slice = slice('2000-01-01', '2100-12-31')
#
# x_train = cmip[final_train_slice]['t2m'].squeeze()
# y_train = t_corr_bc[final_train_slice]['t2m'].squeeze()
# x_predict = cmip[final_predict_slice]['t2m'].squeeze()
#
# bc_cmip = BiasCorrection(y_train, x_train, x_predict)
# t_corr_bc_cmip = pd.DataFrame(bc_cmip.correct(method='normal_correction'))



# WARUM IST t_corr_bc_cmip SO VIEL KÄLTER???

## Step 2 - Downscale CMIP6 using fitted ERA5

# Test for most suitable downscaling algorithm:

train_slice = slice('2000-01-01', '2009-12-31')         # For best algorithm.
predict_slice = slice('2010-01-01', '2020-12-31')       # For best algorithm.
final_train_slice = slice('1982-01-01', '2019-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-31')
plot_slice = slice('2010-01-01', '2020-12-31')

# freq = 'D'
# sds.overview_plot(cmip[plot_slice]['t2m'].resample(freq).mean(), t_corr[plot_slice]['t2m'].resample(freq).mean(),
#                   labelvar1='Temperature [K]')

x_train = cmip[train_slice].drop(columns=['tp'])
y_train = t_corr[train_slice]
x_predict = cmip[predict_slice].drop(columns=['tp'])
y_predict = t_corr[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict)
# sds.modcomp_plot(t_corr_D[plot_slice]['t2m'], x_predict[plot_slice]['t2m'], prediction['predictions'][plot_slice], ylabel='Temperature [K]')
# sds.dmod_score(prediction['predictions'], t_corr_D['t2m'], y_predict['t2m'], x_predict['t2m'])


# Apply best model on full training and prediction periods

x_train = cmip[final_train_slice].drop(columns=['tp'])
y_train = t_corr[final_train_slice]
x_predict = cmip[final_predict_slice].drop(columns=['tp'])
y_predict = t_corr[final_predict_slice]

best_mod = prediction['models']['BCSD: BcsdTemperature']          # GARD: LinearRegression GARD: PureAnalog-best-1 --> better fit, less trend
best_mod.fit(x_train, y_train)
t_corr_cmip = pd.DataFrame(index=x_predict.index)
t_corr_cmip['t2m'] = best_mod.predict(x_predict)



## Check results:

# ERA:
t = slice('1982-01-01', '2020-12-31')
freq = 'Y'
fig, ax = plt.subplots(figsize=(12, 8))
t_corr['t2m'].resample(freq).mean().plot(ax=ax, label='era_bcsd', legend=True)
t_corr_bc['t2m'].resample(freq).mean().plot(ax=ax, label='era_sdm', legend=True)
era_temp_D[t]['t2m'].resample(freq).mean().plot(label='era', ax=ax, legend=True)
aws_temp_D[t]['t2m'].resample(freq).mean().plot(label='aws', ax=ax, legend=True)
plt.show()


compare = pd.concat({'era_bcsd': t_corr['t2m'][t], 'era_sdm': t_corr_bc['t2m'][t],
                     'era': era_temp_D['t2m'][t], 'aws': aws_temp_D[t]['t2m']}, axis=1)
compare.describe()


# CMIP:

t = slice('2000-01-01', '2100-12-31')
freq = 'Y'
fig, ax = plt.subplots(figsize=(12, 8))
# t_corr_cmip['t2m'].resample(freq).mean().plot(ax=ax, label='cmip6_bcsd', legend=True)
t_corr_bc_cmip['t2m'].resample(freq).mean().plot(ax=ax, label='cmip_sdm (result)', legend=True)
cmip[t]['t2m'].resample(freq).mean().plot(label='cmip6 (scen)', ax=ax, legend=True)
# t_corr[t]['t2m'].resample(freq).mean().plot(label='era_bcsd', ax=ax, legend=True)
t_corr_bc[t]['t2m'].resample(freq).mean().plot(label='era_sdm (obs)', ax=ax, legend=True)

plt.show()

compare = pd.concat({'cmip6fitted': t_corr_cmip['t2m'][final_train_slice], 'cmip6': x_predict['t2m'][final_train_slice],
                     'era5fitted': y_predict['t2m'][final_train_slice]}, axis=1)
compare.describe()



freq = 'Y'
time = slice('1982-01-01', '2100-12-31')
fig, ax = plt.subplots(figsize=(12, 8))
trendline(t_corr_cmip['t2m'][time].resample(freq).sum())
t_corr_cmip['t2m'][time].resample(freq).sum().plot(ax=ax, label='cmip-fit', legend=True)
trendline(cmip['t2m'][time].resample(freq).sum())
cmip['t2m'][time].resample(freq).sum().plot(label='cmip-orig', ax=ax, legend=True)
plt.show()


# Check weird peaks in downscaled data:

t = slice('2000-01-01', '2020-12-30')
era_annual = daily_annual_T(era, t)
t_corr_annual = daily_annual_T(t_corr, t)
fig, ax = plt.subplots()
era_annual['t2m'].plot(label='original', ax=ax, legend=True)
t_corr_annual.plot(label='fitted', ax=ax, legend=True, c='red')
plt.show()

t = slice('2000-01-01', '2100-12-30')
cmip_annual = daily_annual_T(cmip, t)
t_corr_annual = daily_annual_T(t_corr_cmip, t)
fig, ax = plt.subplots()
cmip_annual['t2m'].plot(label='original', ax=ax, legend=True)
t_corr_annual['t2m'].plot(label='fitted', ax=ax, legend=True, c='red')
plt.show()









#################################
#   Downscaling precipitation   #
#################################

## Step 1 - Downscale ERA5 using AWS

# Test for most suitable downscaling algorithm:

train_slice = slice('2007-08-01', '2010-12-31')         # For best algorithm.
predict_slice = slice('2011-01-01', '2014-12-31')       # For best algorithm.
final_train_slice = slice('2007-08-01', '2014-12-31')
final_predict_slice = slice('1982-01-01', '2020-12-31')
plot_slice = slice('2011-01-01', '2014-12-31')

# sds.overview_plot(era_D[plot_slice]['tp'], aws_prec[plot_slice]['tp'],
#                   labelvar1='Daily Precipitation [m]', figsize=(15,6))

x_train = era_D[train_slice].drop(columns=['t2m'])
y_train = aws_prec[train_slice]
x_predict = era_D[predict_slice].drop(columns=['t2m'])
y_predict = aws_prec[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict, precip=True, **{'detrend': False})
# sds.modcomp_plot(aws_prec[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice],
#                  ylabel='Daily Precipitation [m]')
# sds.dmod_score(prediction['predictions'], aws_prec['tp'], y_predict['tp'], x_predict['tp'])

# # Compare results with training and target data:
# compare = pd.concat([prediction['predictions'], y_predict['tp']], axis=1)
# comp_desc = compare.describe()
# compare.sum()


# Apply best model on full training and prediction periods

x_train = era_D[final_train_slice].drop(columns=['t2m'])
y_train = aws_prec[final_train_slice]
x_predict = era_D[final_predict_slice].drop(columns=['t2m'])
y_predict = aws_prec[final_predict_slice]

# sds.overview_plot(era_D[final_train_slice]['tp'], aws_prec[final_train_slice]['tp'],
#                   labelvar1='Daily Precipitation [m]')

best_mod = prediction['models']['BCSD: BcsdPrecipitation']
best_mod.fit(x_train, y_train)
p_corr = pd.DataFrame(index=x_predict.index)
p_corr['tp'] = best_mod.predict(x_predict)         # insert scenario data here

# Compare results with training and target data:
# freq = 'W'
# fig, ax = plt.subplots(figsize=(12, 8))
# # x_predict['tp'][final_train_slice].resample(freq).sum().plot(label='era5', ax=ax, legend=True)
# y_predict['tp'].resample(freq).sum().plot(label='aws_prec', ax=ax, legend=True)
# p_corr['tp'][final_train_slice].resample(freq).sum().plot(ax=ax, label='fitted', legend=True)

# compare = pd.concat({'fitted':p_corr['tp'][final_train_slice], 'era5': x_predict['tp'][final_train_slice],
#                      'aws_prec':y_predict['tp'][final_train_slice]}, axis=1)
# compare.describe()
# compare.sum()



## Step 2 - Downscale CMIP6 using fitted ERA5

# Test for most suitable downscaling algorithm:

train_slice = slice('2000-01-01', '2009-12-31')         # For best algorithm.
predict_slice = slice('2010-01-01', '2019-12-31')       # For best algorithm.
final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-30')
plot_slice = slice('2010-01-01', '2019-12-31')

# sds.overview_plot(cmip[plot_slice]['tp'], p_corr[plot_slice]['tp'],
#                   labelvar1='Daily Precipitation [m]')

x_train = cmip[train_slice].drop(columns=['t2m'])
y_train = p_corr[train_slice]
x_predict = cmip[predict_slice].drop(columns=['t2m'])
y_predict = p_corr[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict, precip=True)
# sds.modcomp_plot(p_corr[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice], ylabel='Daily Precipitation [m]')
# sds.dmod_score(prediction['predictions'], p_corr['tp'], y_predict['tp'], x_predict['tp'])

# # Compare results with training and target data:
# compare = pd.concat([prediction['predictions'], y_predict['tp']], axis=1)
# comp_desc = compare.describe()
# compare.sum()


# Apply best model on full training and prediction periods

x_train = cmip[final_train_slice].drop(columns=['t2m'])
y_train = p_corr[final_train_slice]
x_predict = cmip[final_predict_slice].drop(columns=['t2m'])
y_predict = p_corr[final_predict_slice]

best_mod = prediction['models']['BCSD: BcsdPrecipitation']          # Or: GARD: PureAnalog-sample-10
best_mod.fit(x_train, y_train)
p_corr_cmip = pd.DataFrame(index=x_predict.index)
p_corr_cmip['tp'] = best_mod.predict(x_predict)


## Check results:
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


freq = 'Y'
time = slice('1982-01-01', '2100-12-31')
fig, ax = plt.subplots(figsize=(12, 8))
trendline(p_corr_cmip['prec'][time].resample(freq).sum())
p_corr_cmip['prec'][time].resample(freq).sum().plot(ax=ax, label='cmip-fit_', legend=True)
trendline(cmip_prec['prec'][time].resample(freq).sum())
cmip_prec['prec'][time].resample(freq).sum().plot(label='cmip-orig_', ax=ax, legend=True)
plt.show()





################################
#   Saving final time series   #
################################

# cmip_corr = pd.concat([t_corr_cmip, p_corr_cmip], axis=1)
# era_corr = pd.concat([t_corr, p_corr], axis=1)
# cmip_corr.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/CMIP6/'
#                  + 'kyzylsuu_CMIP6_ssp2_4_5_mean_2000_2100_42.25-78.25_fitted2ERA5Lfit.csv')
# era_corr.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/' +
#                 'kyzylsuu_ERA5_Land_1982_2020_42.2_78.2_fitted2AWS.csv')





