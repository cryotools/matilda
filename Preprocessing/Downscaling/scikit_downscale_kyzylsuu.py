##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import salem
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
wd = home + '/Ana-Lena_Phillip/data/scripts/Preprocessing/Downscaling'
import os
os.chdir(wd)
sys.path.append(wd)
import scikit_downscale_matilda as sds
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from skdownscale.pointwise_models import PureAnalog, AnalogRegression
from skdownscale.pointwise_models import BcsdTemperature, BcsdPrecipitation

# interactive plotting?
# plt.ion()

##########################
#   Data preparation:    #
##########################

## ERA5L Gridpoint:

# Apply '/Ana-Lena_Phillip/data/scripts/Tools/ERA5_Subset_Routine.sh' for ncdf-subsetting

in_file = home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/t2m_tp_ERA5L_kyzylsuu_1982_2019.nc'
ds = xr.open_dataset(in_file)
pick = ds.sel(latitude=42.191433, longitude=78.200253, method='nearest')           # closest to AWS location
# pick = pick.sel(time=slice('1989-01-01', '2019-12-31'))                      # start of gauging till end of file
era = pick.to_dataframe().filter(['t2m', 'tp'])
era['tp'][era['tp'] < 0.0001] = 0                                                # Negative values in the data
# era.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/t2m_tp_ERA5L_kyzylsuu_42.2_78.2_1982_2019.csv')

era_D = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})

## AWS Chong Kyzylsuu:
aws_temp = pd.read_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/' +
                       'temp_kyzylsuu_2007-2015.csv',
                       parse_dates=['time'], index_col='time')
aws_temp['t2m'] = aws_temp['t2m'] + 273.15
aws_temp_D = aws_temp.resample('D').mean()                # Precipitation data is daily!

aws_prec = pd.read_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/' +
                       'prec_kyzylsuu_2007-2014.csv',
                       parse_dates=['time'], index_col='time')
aws_prec['tp'] = aws_prec['tp']/1000


aws = pd.merge(aws_temp_D, aws_prec, how='outer', left_index=True, right_index=True)
# aws.to_csv('/home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Weather station/' +
# 'met_data_full_kyzylsuu_2007-2015.csv')

# temp data gaps: 2010-03-30' to '2010-04-01', '2011-10-12' to '2011-10-31', '2014-05-02' to '2014-05-03',
# '2015-06-05' to '2015-06-06'


## CMIP6:

cmip = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/' +
                       'CMIP6_mean_42.25-78.25_2000-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip = cmip.filter(like='_45')              # To select scenario e.g. RCP4.5 from the model means
# cmip = cmip.tz_localize('UTC')
cmip.columns = era.columns
cmip['tp'] = cmip['tp']/1000
cmip = cmip.resample('D').mean()        # Already daily but wrong daytime (12:00:00).




## Overview

# prec: 2007-08-01 to 2014-12-31
# temp: 2007-08-10 to 2016-01-01
# AWS location: 42.191433, 78.200253
# Gauging station Hydromet: 1989-01-01 to 2019-09-09
# CMIP: 2000-01-01 to 2100-12-31
# ERA: 1982-01-01 to 2019-12-31

# t = slice('2007-08-10', '2014-12-31')
# d = {'AWS': aws[t]['t2m'], 'ERA5L': era_D[t]['t2m'], 'CMIP6': cmip[t]['t2m']}
# data = pd.DataFrame(d)
# data.describe()
# data.plot(figsize=(12, 6))
# plt.show()

t = slice('2007-08-10', '2014-12-31')
d = {'ERA5': era[t]['tp'], 'AWS': aws_prec[t]['tp'], 'CMIP6': cmip[t]['tp']}
data = pd.DataFrame(d)
data.resample('M').sum()
data.resample('M').sum().plot(figsize=(12, 6))
# data.describe()
plt.show()

#################################
#    Downscaling temperature    #
#################################

## Step 1 - Downscale ERA5 using AWS

# Apply BCSD-Temperature right away because it can cope with NAs

final_train_slice = slice('2007-08-10', '2016-01-01')
final_predict_slice = slice('2000-01-01', '2019-12-31')
plot_slice = slice('2007-08-10', '2016-01-01')

# sds.overview_plot(era[plot_slice]['t2m'], aws_temp[plot_slice]['t2m'],
#                   labelvar1='Temperature [K]')

x_train = era[final_train_slice].drop(columns=['tp'])
y_train = aws_temp[final_train_slice]
x_predict = era[final_predict_slice].drop(columns=['tp'])
y_predict = aws_temp[final_predict_slice]

fit = BcsdTemperature(return_anoms=False).fit(x_train, y_train)
t_corr = pd.DataFrame(index=x_predict.index)
t_corr['t2m'] = fit.predict(x_predict)
t_corr_D = t_corr.resample('D').mean()

# freq = 'M'
# fig, ax = plt.subplots(figsize=(12,8))
# t_corr['t2m'][plot_slice].resample(freq).mean().plot(ax=ax, label='fitted', legend=True)
# era['t2m'][plot_slice].resample(freq).mean().plot(label='era5', ax=ax, legend=True)
# aws_temp['t2m'][plot_slice].resample(freq).mean().plot(label='aws', ax=ax, legend=True)



## Step 2 - Downscale CMIP6 using fitted ERA5

# Test for most suitable downscaling algorithm:

train_slice = slice('2000-01-01', '2009-12-31')         # For best algorithm.
predict_slice = slice('2010-01-01', '2019-12-31')       # For best algorithm.
final_train_slice = slice('2000-01-01', '2019-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-31')
plot_slice = slice('2010-01-01', '2019-12-31')

freq = 'D'
sds.overview_plot(cmip[plot_slice]['t2m'].resample(freq).mean(), t_corr[plot_slice]['t2m'].resample(freq).mean(),
                  labelvar1='Temperature [K]')

x_train = cmip[train_slice].drop(columns=['tp'])
y_train = t_corr_D[train_slice]
x_predict = cmip[predict_slice].drop(columns=['tp'])
y_predict = t_corr_D[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict)
sds.modcomp_plot(t_corr_D[plot_slice]['t2m'], x_predict[plot_slice]['t2m'], prediction['predictions'][plot_slice], ylabel='Temperature [K]')
sds.dmod_score(prediction['predictions'], t_corr_D['t2m'], y_predict['t2m'], x_predict['t2m'])


# Apply best model on full training and prediction periods

x_train = cmip[final_train_slice].drop(columns=['tp'])
y_train = t_corr_D[final_train_slice]
x_predict = cmip[final_predict_slice].drop(columns=['tp'])
y_predict = t_corr_D[final_predict_slice]

best_mod = prediction['models']['BCSD: BcsdTemperature']          # GARD: PureAnalog-best-1 --> better fit, less trend
best_mod.fit(x_train, y_train)
t_corr_cmip = pd.DataFrame(index=x_predict.index)
t_corr_cmip['t2m'] = best_mod.predict(x_predict)


# # Compare results with training and target data:
freq = 'Y'
fig, ax = plt.subplots(figsize=(12, 8))
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

train_slice = slice('2007-08-01', '2010-12-31')         # For best algorithm.
predict_slice = slice('2011-01-01', '2014-12-31')       # For best algorithm.
final_train_slice = slice('2007-08-01', '2014-12-31')
final_predict_slice = slice('2000-01-01', '2019-12-31')
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

sds.overview_plot(era_D[final_train_slice]['tp'], aws_prec[final_train_slice]['tp'],
                  labelvar1='Daily Precipitation [m]')

best_mod = prediction['models']['BCSD: BcsdPrecipitation']
best_mod.fit(x_train, y_train)
p_corr = pd.DataFrame(index=x_predict.index)
p_corr['tp'] = best_mod.predict(x_predict)         # insert scenario data here

# Compare results with training and target data:
# freq = 'Y'
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
final_train_slice = slice('2000-01-01', '2019-12-31')
final_predict_slice = slice('2000-01-01', '2100-12-30')
plot_slice = slice('2010-01-01', '2019-12-31')

sds.overview_plot(cmip[plot_slice]['tp'], p_corr[plot_slice]['tp'],
                  labelvar1='Daily Precipitation [m]')

x_train = cmip[train_slice].drop(columns=['t2m'])
y_train = p_corr[train_slice]
x_predict = cmip[predict_slice].drop(columns=['t2m'])
y_predict = p_corr[predict_slice]

prediction = sds.fit_dmodels(x_train, y_train, x_predict, precip=True)
sds.modcomp_plot(p_corr[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice], ylabel='Daily Precipitation [m]')
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
era_corr = pd.concat([t_corr_D, p_corr], axis=1)
cmip_corr.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/CMIP6/'
                 + 'kyzylsuu_CMIP6_ssp2_4_5_mean_2000_2100_42.25-78.25_fitted2ERA5Lfit.csv')
era_corr.to_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/' +
                'kyzylsuu_ERA5_Land_2000_2019_42.2_78.2_fitted2AWS.csv')


