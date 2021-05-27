##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
home = str(Path.home()) + '/Seafile/Ana-Lena_Phillip/data'
wd = home + '/scripts/Test_area/Scikit-downscale'
import os
os.chdir(wd)
sys.path.append(wd)
from utils import get_sample_data
sns.set(style='darkgrid')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from skdownscale.pointwise_models import PureAnalog, AnalogRegression
from skdownscale.pointwise_models import BcsdTemperature, BcsdPrecipitation
from utils import prob_plots

# interactive plotting?
plt.ion()

## load my data

dat = pd.read_csv('/home/phillip/Seafile/EBA-CA/Tianshan_data/AWS_atbs/atbs_met-data_2017-2020.csv',
                  parse_dates=['datetime'], index_col='datetime')
# dat = dat.resample('D').mean()

training = dat.drop(columns=['temp_era_fitted', 'temp_minikin'])
targets = dat.drop(columns=['rh', 'ws', 'wd'])          # HIER MIT ERA5L-DATEN FORTFAHREN!!

train_slice = slice('2018-09-07', '2019-03-13')
predict_slice = slice('2019-03-14', '2019-09-13')

## example
# dat = pd.read_csv(home + "/input_output/input/ERA5/Tien-Shan/At-Bashy/Bash-Kaindy_preprocessed_forcing_data.csv",
#                   parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
# # dat = dat.resample('D').mean()
#
# training = dat.drop(columns=['temp_era_fitted', 'temp_minikin'])
# targets = dat.drop(columns=['temp_era_fitted', 'temp_era'])
#
# train_slice = slice('2018-09-07', '2019-09-13')
# predict_slice = slice('2018-09-07', '2019-09-13')

# print a table of the training/targets data
# display(pd.concat({'training': training, 'targets': targets}, axis=1))

## plot temperature and precipitation data
plot_slice = slice('2018-09-07', '2019-09-13')

fig, axes = plt.subplots()
axes.plot(training[plot_slice], label='training')
axes.plot(targets[plot_slice], label='targets')
axes.legend()
axes.set_ylabel('Temperature [C]')

##
models = {
    'GARD: PureAnalog-best-1': PureAnalog(kind='best_analog', n_analogs=1),
    'GARD: PureAnalog-sample-10': PureAnalog(kind='sample_analogs', n_analogs=10),
    'GARD: PureAnalog-weight-10': PureAnalog(kind='weight_analogs', n_analogs=10),
    'GARD: PureAnalog-weight-100': PureAnalog(kind='weight_analogs', n_analogs=100),
    'GARD: PureAnalog-mean-10': PureAnalog(kind='mean_analogs', n_analogs=10),
    'GARD: AnalogRegression-100': AnalogRegression(n_analogs=100),
    'GARD: LinearRegression': LinearRegression(),
    'BCSD: BcsdTemperature': BcsdTemperature(return_anoms=False),             # Only works with decent training period.
    'Sklearn: RandomForestRegressor': RandomForestRegressor(random_state=0)
}

# extract training / prediction data
X_train = training[train_slice]
y_train = targets[train_slice]
X_predict = training[predict_slice]

##
# Fit all models
for key, model in models.items():
    model.fit(X_train, y_train)

# store predicted results in this dataframe
predict_df = pd.DataFrame(index = X_predict.index)

for key, model in models.items():
    predict_df[key] = model.predict(X_predict)

# # show a table of the predicted data
# display(predict_df.head())

##
fig, ax = plt.subplots(figsize=(10,5))
targets[plot_slice].plot(ax=ax, label='target', c='k', lw=1, alpha=0.75, legend=True, zorder=10)
X_predict[plot_slice].plot(label='original', c='grey', ax=ax, alpha=0.75, legend=True)
predict_df[plot_slice].plot(ax=ax, lw=0.75)
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
_ = ax.set_ylabel('Temperature [C]')
# fig.savefig('model_comparison', bbox_inches='tight')

plt.subplots_adjust(right=0.66, bottom=0.15, top=0.95)      # Passt sonst nicht ganz drauf.
##
# calculate r2
score = (predict_df.corrwith(targets.temp_minikin[predict_slice]) **2).sort_values().to_frame('r2_score')
display(score)

fig = prob_plots(X_predict, targets['temp_minikin'], predict_df[score.index.values], shape=(3, 3), figsize=(12, 12))
