##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
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
wd = home + '/Ana-Lena_Phillip/data/scripts/Preprocessing/Downscaling'
import os
os.chdir(wd)
sys.path.append(wd)
import scikit_downscale_matilda as sds


# interactive plotting?
# plt.ion()

##########################
#   Data preparation:    #
##########################

# ERA5 closest gridpoint:
era = pd.read_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy'
                           '/no182ERA5_Land_lat41.0_lon75.9_alt3839.6_1981-2019.csv', index_col='time')
era.index = pd.to_datetime(era.index, format='%d.%m.%Y %H:%M')
era = era.tz_localize('UTC')
era = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})


# AWS Bash Kaingdy:
aws = pd.read_csv(home + '/EBA-CA/Tianshan_data/AWS_atbs/atbs_met-data_2017-2020.csv',
                  parse_dates=['datetime'], index_col='datetime')
aws = aws.shift(periods=6, freq="H")                                     # Data is still not aligned with UTC
aws = aws.tz_convert('UTC')
aws = aws.drop(columns=['rh', 'ws', 'wd'])                    # Need to be dataframes not series!
aws.columns = era.columns
    # Downscaling cannot cope with data gaps:
aws = aws.resample('D').agg({'t2m': 'mean', 'tp':'sum'})
aws = aws.interpolate(method='spline', order=2)           # No larger data gaps after 2017-07-04


# Minikin-data:
minikin = pd.read_csv(home + "/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy" +
                             "/Bash-Kaindy_preprocessed_forcing_data.csv",
                      parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
minikin = minikin.filter(like='_minikin')
minikin.columns = ['t2m']
minikin = minikin.resample('D').mean()


# CMIP6 data Bash Kaingdy:
cmip = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/' +
                       'ssp2_4_5_41-75.9_2000-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip = cmip.filter(like='_mean')
cmip = cmip.tz_localize('UTC')
cmip.columns = era.columns
cmip = cmip.resample('D').mean()        # Already daily but wrong daytime (12:00:00 --> lesser days overall).
cmip = cmip.interpolate(method='spline', order=2)       # Only 25 days in 100 years, only 3 in fitting period.


## Overview

# aws:      2017-07-14 to 2020-09-30    --> Available from 2017-06-02 but biggest datagap: 2017-07-04 to 2017-07-14
# era:      1981-01-01 to 2019-12-31
# minikin:  2018-09-07 to 2019-09-13
# cmip:     2000-01-01 to 2100-12-30

# t = slice('2018-09-07', '2019-09-13')
# d = {'AWS': aws[t]['t2m'], 'ERA5': era[t]['t2m'], 'Minikin': minikin[t]['t2m'], 'CMIP6': cmip[t]['t2m']}
# data = pd.DataFrame(d)
# data.plot(figsize=(12, 6))


#################################
#    Downscaling temperature    #
#################################

## Step 1 - Downscale ERA5 using AWS

# Test for most suitable downscaling algorithm:

train_slice = slice('2017-07-14', '2018-09-30')         # For best algorithm.
predict_slice = slice('2018-09-30', '2019-12-31')       # For best algorithm.
final_train_slice = slice('2017-07-14', '2019-12-31')
final_predict_slice = slice('2000-01-01', '2019-12-31')
plot_slice = slice('2017-07-14', '2019-12-31')

sds.overview_plot(era[plot_slice]['t2m'], aws[plot_slice]['t2m'],
                  labelvar1='Temperature [°C]')

x_train = era[train_slice].drop(columns=['tp'])
y_train = aws[train_slice].drop(columns=['tp'])
x_predict = era[predict_slice].drop(columns=['tp'])
y_predict = aws[predict_slice].drop(columns=['tp'])

prediction = sds.fit_dmodels(x_train, y_train, x_predict)
sds.modcomp_plot(aws[predict_slice]['t2m'], x_predict[predict_slice]['t2m'], prediction['predictions'][predict_slice], ylabel='Temperature [°C]')
sds.dmod_score(prediction['predictions'], aws['t2m'], y_predict['t2m'], x_predict['t2m'])


# Apply best model on full training and prediction periods

x_train = era[final_train_slice].drop(columns=['tp'])
y_train = aws[final_train_slice].drop(columns=['tp'])
x_predict = era[final_predict_slice].drop(columns=['tp'])
y_predict = aws[final_predict_slice].drop(columns=['tp'])

best_mod = prediction['models']['BCSD: BcsdTemperature']          # Pick the best model by name.
best_mod.fit(x_train, y_train)
t_corr = pd.DataFrame(index=x_predict.index)
t_corr['t2m'] = best_mod.predict(x_predict)         # insert scenario data here


# # Compare results with training and target data:

fig, ax = plt.subplots(figsize=(12,8))
t_corr['t2m'][final_train_slice].plot(ax=ax, label='fitted', legend=True)
x_predict['t2m'][final_train_slice].plot(label='era5', ax=ax, legend=True)
y_predict['t2m'].plot(label='aws', ax=ax, legend=True)

# compare = pd.concat({'fitted':t_corr['t2m'][final_train_slice], 'era5': x_predict['t2m'][final_train_slice],
#                      'aws':y_predict['t2m'][final_train_slice]}, axis=1)
# compare.describe()


## Step 2 - Downscale CMIP6 using fitted ERA5

# Test for most suitable downscaling algorithm:

train_slice = slice('2000-01-01', '2009-12-31')         # For best algorithm.
predict_slice = slice('2010-01-01', '2019-12-31')       # For best algorithm.
final_train_slice = slice('2000-01-01', '2019-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-30')
plot_slice = slice('2010-01-01', '2019-12-31')

sds.overview_plot(cmip[plot_slice]['t2m'], t_corr[plot_slice]['t2m'],
                  labelvar1='Temperature [°C]')

x_train = cmip[train_slice].drop(columns=['tp'])
y_train = t_corr[train_slice]
x_predict = cmip[predict_slice].drop(columns=['tp'])
y_predict = t_corr[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict)
sds.modcomp_plot(t_corr[plot_slice]['t2m'], x_predict[plot_slice]['t2m'], prediction['predictions'][plot_slice], ylabel='Temperature [°C]')
sds.dmod_score(prediction['predictions'], t_corr['t2m'], y_predict['t2m'], x_predict['t2m'])


# Apply best model on full training and prediction periods

x_train = cmip[final_train_slice].drop(columns=['tp'])
y_train = t_corr[final_train_slice]
x_predict = cmip[final_predict_slice].drop(columns=['tp'])
y_predict = t_corr[final_predict_slice]

best_mod = prediction['models']['GARD: PureAnalog-weight-10']          # Pick the best model by name.
best_mod.fit(x_train, y_train)
t_corr_cmip = pd.DataFrame(index=x_predict.index)
t_corr_cmip['t2m'] = best_mod.predict(x_predict)


# # Compare results with training and target data:
freq = 'Y'
fig, ax = plt.subplots(figsize=(12,8))
t_corr_cmip['t2m'].resample(freq).mean().plot(ax=ax, label='cmip6_fitted', legend=True)
x_predict['t2m'].resample(freq).mean().plot(label='cmip6', ax=ax, legend=True)
y_predict['t2m'].resample(freq).mean().plot(label='era5l_fitted', ax=ax, legend=True)

# compare = pd.concat({'cmip6fitted': t_corr_cmip['t2m'][final_train_slice], 'cmip6': x_predict['t2m'][final_train_slice],
#                      'era5fitted': y_predict['t2m'][final_train_slice]}, axis=1)
# compare.describe()



#################################
#   Downscaling precipitation   #
#################################

## Step 1 - Downscale ERA5 using AWS

# Test for most suitable downscaling algorithm:

train_slice = slice('2017-07-14', '2018-09-30')         # For best algorithm.
predict_slice = slice('2018-09-30', '2019-12-31')       # For best algorithm.
final_train_slice = slice('2017-07-14', '2019-12-31')   # 2020-04-25: Extreme outlier on 2020-04-26, ~0 Precip after that!
final_predict_slice = slice('2000-01-01', '2019-12-31')
plot_slice = slice('2018-09-30', '2019-12-31')

sds.overview_plot(era[plot_slice]['tp'], aws[plot_slice]['tp'],
                  labelvar1='Daily Precipitation [mm]', figsize=(15,6))

x_train = era[train_slice].drop(columns=['t2m'])
y_train = aws[train_slice].drop(columns=['t2m'])
x_predict = era[predict_slice].drop(columns=['t2m'])
y_predict = aws[predict_slice].drop(columns=['t2m'])

prediction = sds.fit_dmodels(x_train, y_train, x_predict, precip=True,
                             **{'detrend': False})                          # Detrending results in negative values.
sds.modcomp_plot(aws[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice],
                 ylabel='Daily Precipitation [mm]')
sds.dmod_score(prediction['predictions'], aws['tp'], y_predict['tp'], x_predict['tp'])


# # Compare results with training and target data:
# compare = pd.concat([prediction['predictions'], y_predict['tp']], axis=1)
# comp_desc = compare.describe()
# compare.sum()                           # PRECIPITATION GETS A LOT MORE IN EVERY CASE!


# Apply best model on full training and prediction periods

x_train = era[final_train_slice].drop(columns=['t2m'])
y_train = aws[final_train_slice].drop(columns=['t2m'])
x_predict = era[final_predict_slice].drop(columns=['t2m'])
y_predict = aws[final_predict_slice].drop(columns=['t2m'])

sds.overview_plot(era[final_train_slice]['tp'], aws[final_train_slice]['tp'],
                  labelvar1='Daily Precipitation [mm]')

best_mod = prediction['models']['BCSD: BcsdPrecipitation']          # Pick the best model by name.
best_mod.fit(x_train, y_train)
p_corr = pd.DataFrame(index=x_predict.index)
p_corr['tp'] = best_mod.predict(x_predict)         # insert scenario data here

# Compare results with training and target data:

fig, ax = plt.subplots(figsize=(12,8))
p_corr['tp'][final_train_slice].plot(ax=ax, label='fitted', legend=True)
x_predict['tp'][final_train_slice].plot(label='era5', ax=ax, legend=True)
y_predict['tp'].plot(label='aws', ax=ax, legend=True)

# compare = pd.concat({'fitted':p_corr['tp'][final_train_slice], 'era5': x_predict['tp'][final_train_slice],
#                      'aws':y_predict['tp'][final_train_slice]}, axis=1)
# compare.describe()
# compare.sum()                       # Here the precipitation gets less than the observed...

# KÖNNTE BESSER AUSSEHEN....HAR-DATEN VERWENDEN?



## Step 2 - Downscale CMIP6 using fitted ERA5

# Test for most suitable downscaling algorithm:

train_slice = slice('2000-01-01', '2009-12-31')         # For best algorithm.
predict_slice = slice('2010-01-01', '2019-12-31')       # For best algorithm.
final_train_slice = slice('2000-01-01', '2019-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-30')
plot_slice = slice('2010-01-01', '2019-12-31')

sds.overview_plot(cmip[plot_slice]['tp'], p_corr[plot_slice]['tp'],
                  labelvar1='Daily Precipitation [mm]')

x_train = cmip[train_slice].drop(columns=['t2m'])
y_train = p_corr[train_slice]
x_predict = cmip[predict_slice].drop(columns=['t2m'])
y_predict = p_corr[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict, precip=True)
sds.modcomp_plot(p_corr[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice], ylabel='Daily Precipitation [mm]')
sds.dmod_score(prediction['predictions'], p_corr['tp'], y_predict['tp'], x_predict['tp'])

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


# Compare results with training and target data:
freq = 'Y'
fig, ax = plt.subplots(figsize=(12, 8))
p_corr_cmip['tp'].resample(freq).mean().plot(ax=ax, label='cmip6_fitted', legend=True)
x_predict['tp'].resample(freq).mean().plot(label='cmip6', ax=ax, legend=True)
y_predict['tp'].resample(freq).mean().plot(label='era5l_fitted', ax=ax, legend=True)

# compare = pd.concat({'cmip6fitted':p_corr_cmip['tp'][final_train_slice], 'cmip6': x_predict['tp'][final_train_slice],
#                      'era5fitted':y_predict['tp'][final_train_slice]}, axis=1)
# compare.describe()
# compare.sum()



################################
#   Saving final time series   #
################################

cmip_corr = pd.concat([t_corr_cmip, p_corr_cmip], axis=1)
era_corr = pd.concat([t_corr, p_corr], axis=1)
cmip_corr.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/CMIP6/'
                 + 'no182_CMIP6_ssp2_4_5_mean_2000_2100_41-75.9_fitted2ERA5Lfit.csv')
era_corr.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/' +
                'no182_ERA5_Land_2000_2019_41_75.9_fitted2AWS.csv')






# #########################################################################################################
# #
# train_slice = slice('2017-07-14', '2019-12-31')
# predict_slice = slice('2017-07-14', '2019-12-31')
# # predict_slice = slice('1981-01-01', '2019-12-31')
# plot_slice = slice('2017-07-14', '2019-12-31')
#
#
# ## Plot both datasets
# sds.overview_plot(training[plot_slice]['t2m'], targets[plot_slice]['t2m'],
#                   training[plot_slice]['tp'], targets[plot_slice]['tp'], no_var=2,
#                   labelvar1='Temperature [°C]', labelvar2='Precipitation [mm]')
#
# ## extract training / prediction data
# x_train = training[train_slice]
# y_train = targets[train_slice]
# x_predict = training[predict_slice]
# y_predict = targets[predict_slice]      # FRAGLICH, WEIL ZU LANG! Only for finding the best model.
#
# # For temperature:
# x_train_t = x_train.drop(columns=['tp'])                            # Need to be dataframes not series!
# y_train_t = y_train.drop(columns=['tp'])
# x_predict_t = x_predict.drop(columns=['tp'])
# y_predict_t = y_predict.drop(columns=['tp'])
#
#
# ## Fit models and predict data series:
#
# prediction = sds.fit_dmodels(x_train_t, y_train_t, x_predict_t)
#
# sds.modcomp_plot(targets[plot_slice]['t2m'], x_predict[plot_slice]['t2m'], prediction['predictions'][plot_slice], ylabel='Temperatur [°C]')
# # sds.dmod_score(prediction['predictions'], targets['t2m'], y_predict['t2m'], x_predict)        # Welcher Zeitraum für y-predict?
# sds.dmod_score(prediction['predictions'], targets['t2m'], targets['t2m'], x_predict['t2m'])
#
# ## Pick best model
#
# best_mod = prediction['models']['GARD: PureAnalog-weight-100']          # Pick the best model by name.
# best_mod.fit(x_train_t, y_train_t)
# t_corr = pd.DataFrame(index=y_predict_t.index)
# t_corr['t_corr'] = best_mod.predict(y_predict_t)         # insert scenario data here
#
# sds.overview_plot(targets['t2m'], y_predict_t,
#                   labelvar1='Temperature [°C]', label_train='before', label_target='after')
#
# ## Apply for precipitation:
#
# x_train_p = x_train.drop(columns=['t2m'])
# y_train_p = y_train.drop(columns=['t2m'])
# x_predict_p = x_predict.drop(columns=['t2m'])
# y_predict_p = y_predict.drop(columns=['t2m'])
#
# prediction = sds.fit_dmodels(x_train_p, y_train_p, x_predict_p, precip=True)
# sds.modcomp_plot(targets[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice], ylabel='Daily Precipitation [mm]')
# sds.dmod_score(prediction['predictions'], targets['tp'], targets['tp'], x_predict['tp'])
# comp = prediction['predictions'].agg(sum)
# comp['obs'] = y_train_p
# comp['era5'] = x_train_p
# comp = comp.describe()
# prediction['predictions'].agg(sum)
#
# best_mod = prediction['models']['GARD: PureAnalog-weight-100']          # Pick the best model by name.
# best_mod = prediction['models']['Sklearn: RandomForestRegressor']
# best_mod = prediction['models']['GARD: PureAnalog-best-1']
# best_mod = prediction['models']['GARD: PureAnalog-weight-10']
#
# best_mod.fit(x_train_p, y_train_p)
# p_corr = pd.DataFrame(index=y_predict_p.index)
# p_corr['p_corr'] = best_mod.predict(y_predict_p)         # insert scenario data here
#
# sds.overview_plot(y_train_p, p_corr['p_corr'],
#                   labelvar1='Daily Precipitation [mm]', label_train='obs', label_target='corrected')


## example
# dat = pd.read_csv(home + "/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/Bash-Kaindy_preprocessed_forcing_data.csv",
#                   parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
# # dat = dat.resample('D').mean()
#
# training = dat.drop(columns=['temp_era_fitted', 'temp_minikin'])        # Need to be dataframes
# targets = dat.drop(columns=['temp_era_fitted', 'temp_era'])             # Need to be dataframes
#
# train_slice = slice('2018-09-07', '2019-09-13')
# predict_slice = slice('2018-09-07', '2019-09-13')
# plot_slice = slice('2018-09-07', '2019-09-13')
#
# prediction = sds.fit_dmodels(x_train, y_train, x_predict)
#
# sds.modcomp_plot(targets[plot_slice], x_predict[plot_slice], prediction['predictions'][plot_slice])
# sds.dmod_score(prediction['predictions'], targets['temp_minikin'], y_predict['temp_minikin'], x_predict)