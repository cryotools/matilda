##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
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


# interactive plotting?
plt.ion()

## load data

# Bash Kaingdy:

# ERA5 closest gridpoint:
training = pd.read_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy'
                           '/no182ERA5_Land_lat41.0_lon75.9_alt3839.6_1981-2019.csv', index_col='time')
training.index = pd.to_datetime(training.index, format='%d.%m.%Y %H:%M')
training = training.tz_localize('UTC')


# AWS Bash Kaingdy:
    # Biggest datagap: 2017-07-04 to 2017-07-14 --> slice('2017-07-14', '2019-12-31')
targets = pd.read_csv(home + '/EBA-CA/Tianshan_data/AWS_atbs/atbs_met-data_2017-2020.csv',
                  parse_dates=['datetime'], index_col='datetime')
targets = targets.shift(periods=6, freq="H")                                     # Data is still not aligned with UTC
targets = targets.tz_convert('UTC')
targets = targets.drop(columns=['rh', 'ws', 'wd'])                    # Need to be dataframes not series!
        # Biggest datagap: 2017-07-04 to 2017-07-14 --> slice('2017-07-14', '2019-12-31')


# Minikin-data:
targets = pd.read_csv(home + "/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/Bash-Kaindy_preprocessed_forcing_data.csv",
                  parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
targets = targets.filter(like='_minikin')


# CMIP6 data Bash Kaingdy:

training = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/' +
                       'ssp2_4_5_41-75.9_2000-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
training = training.filter(like='_mean')
training = training.tz_localize('UTC')

## EINZELNE DATENSÄTZE UMBENENNEN und erst bei x_train etc. zuordnen. CMIP6 MIT Minikin UND AWS-PREC DOWNSCALEN.


##
targets.columns = training.columns                          # Same columns names

train_slice = slice('2017-07-14', '2019-12-31')             # Biggest datagap: 2017-07-04 to 2017-07-14
predict_slice = slice('2017-07-14', '2019-12-31')
# predict_slice = slice('1981-01-01', '2019-12-31')
plot_slice = slice('2017-07-14', '2019-12-31')

## Downscaling cannot cope with data gaps:
targets = targets.resample('D').agg({'t2m': 'mean', 'tp':'sum'})
training = training.resample('D').agg({'t2m': 'mean', 'tp':'sum'})
targets = targets.interpolate(method='spline', order=2)           # No larger datagaps after 2017-07-04

## Plot both datasets
sds.overview_plot(training[plot_slice]['t2m'], targets[plot_slice]['t2m'],
                  training[plot_slice]['tp'], targets[plot_slice]['tp'], no_var=2,
                  labelvar1='Temperature [°C]', labelvar2='Precipitation [mm]')

## extract training / prediction data
x_train = training[train_slice]
y_train = targets[train_slice]
x_predict = training[predict_slice]
y_predict = targets[predict_slice]      # FRAGLICH, WEIL ZU LANG! Only for finding the best model.

# For temperature:
x_train_t = x_train.drop(columns=['tp'])                            # Need to be dataframes not series!
y_train_t = y_train.drop(columns=['tp'])
x_predict_t = x_predict.drop(columns=['tp'])
y_predict_t = y_predict.drop(columns=['tp'])


## Fit models and predict data series:

prediction = sds.fit_dmodels(x_train_t, y_train_t, x_predict_t)

sds.modcomp_plot(targets[plot_slice]['t2m'], x_predict[plot_slice]['t2m'], prediction['predictions'][plot_slice], ylabel='Temperatur [°C]')
# sds.dmod_score(prediction['predictions'], targets['t2m'], y_predict['t2m'], x_predict)        # Welcher Zeitraum für y-predict?
sds.dmod_score(prediction['predictions'], targets['t2m'], targets['t2m'], x_predict['t2m'])

## Pick best model

best_mod = prediction['models']['GARD: PureAnalog-weight-100']          # Pick the best model by name.
best_mod.fit(x_train_t, y_train_t)
t_corr = pd.DataFrame(index=y_predict_t.index)
t_corr['t_corr'] = best_mod.predict(y_predict_t)         # insert scenario data here

sds.overview_plot(targets['t2m'], y_predict_t,
                  labelvar1='Temperature [°C]', label_train='before', label_target='after')

## Apply for precipitation:

x_train_p = x_train.drop(columns=['t2m'])
y_train_p = y_train.drop(columns=['t2m'])
x_predict_p = x_predict.drop(columns=['t2m'])
y_predict_p = y_predict.drop(columns=['t2m'])

prediction = sds.fit_dmodels(x_train_p, y_train_p, x_predict_p, precip=True)
sds.modcomp_plot(targets[plot_slice]['tp'], x_predict[plot_slice]['tp'], prediction['predictions'][plot_slice], ylabel='Daily Precipitation [mm]')
sds.dmod_score(prediction['predictions'], targets['tp'], targets['tp'], x_predict['tp'])
comp = prediction['predictions'].agg(sum)
comp['obs'] = y_train_p
comp['era5'] = x_train_p
comp = comp.describe()
prediction['predictions'].agg(sum)

best_mod = prediction['models']['GARD: PureAnalog-weight-100']          # Pick the best model by name.
best_mod = prediction['models']['Sklearn: RandomForestRegressor']
best_mod = prediction['models']['GARD: PureAnalog-best-1']
best_mod = prediction['models']['GARD: PureAnalog-weight-10']

best_mod.fit(x_train_p, y_train_p)
p_corr = pd.DataFrame(index=y_predict_p.index)
p_corr['p_corr'] = best_mod.predict(y_predict_p)         # insert scenario data here

sds.overview_plot(y_train_p, p_corr['p_corr'],
                  labelvar1='Daily Precipitation [mm]', label_train='obs', label_target='corrected')


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