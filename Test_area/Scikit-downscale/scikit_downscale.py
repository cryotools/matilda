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
from skdownscale.pointwise_models import QuantileMapper
from utils import prob_plots


## load sample data
training = get_sample_data('training')
targets = get_sample_data('targets')

train_slice = slice('1980-01-01', '1989-12-31')
predict_slice = slice('1990-01-01', '1999-12-31')

# print a table of the training/targets data
display(pd.concat({'training': training, 'targets': targets}, axis=1))

# plot temperature and precipitation data
plot_slice = slice('1990-01-01', '1990-12-31')
fig, axes = plt.subplots(ncols=1, nrows=2, figsize=(8, 6), sharex=True)
axes[0].plot(training[plot_slice]['tmax'], label='training')
axes[0].plot(targets[plot_slice]['tmax'], label='targets')
axes[0].legend()
axes[0].set_ylabel('Temperature [C]')

axes[1].plot(training[plot_slice]['pcp'])
axes[1].plot(targets[plot_slice]['pcp'])
_ = axes[1].set_ylabel('Precipitation [mm/day]')

plt.show()

models = {
    'GARD: PureAnalog-best-1': PureAnalog(kind='best_analog', n_analogs=1),
    'GARD: PureAnalog-sample-10': PureAnalog(kind='sample_analogs', n_analogs=10),
    'GARD: PureAnalog-weight-10': PureAnalog(kind='weight_analogs', n_analogs=10),
    'GARD: PureAnalog-weight-100': PureAnalog(kind='weight_analogs', n_analogs=100),
    'GARD: PureAnalog-mean-10': PureAnalog(kind='mean_analogs', n_analogs=10),
    'GARD: AnalogRegression-100': AnalogRegression(n_analogs=100),
    'GARD: LinearRegression': LinearRegression(),
    'BCSD: BcsdTemperature': BcsdTemperature(return_anoms=False),
    'Sklearn: RandomForestRegressor': RandomForestRegressor(random_state=0)
}

# extract training / prediction data
X_train = training[['tmax']][train_slice]
y_train = targets[['tmax']][train_slice]
X_predict = training[['tmax']][predict_slice]

# Fit all models
for key, model in models.items():
    model.fit(X_train, y_train)

# store predicted results in this dataframe
predict_df = pd.DataFrame(index = X_predict.index)

for key, model in models.items():
    predict_df[key] = model.predict(X_predict)

# show a table of the predicted data
display(predict_df.head())

fig, ax = plt.subplots(figsize=(10,5))
targets['tmax'][plot_slice].plot(ax=ax, label='target', c='k', lw=1, alpha=0.75, legend=True, zorder=10)
X_predict['tmax'][plot_slice].plot(label='original', c='grey', ax=ax, alpha=0.75, legend=True)
predict_df[plot_slice].plot(ax=ax, lw=0.75)
ax.legend(loc='center left', bbox_to_anchor=(1,0.5))
_ = ax.set_ylabel('Temperature [C]')
# fig.savefig('model_comparison', bbox_inches='tight')

plt.subplots_adjust(right=0.66, bottom=0.15, top=0.95)      # Passt sonst nicht ganz drauf.
plt.show()

# calculate r2
score = (predict_df.corrwith(targets.tmax[predict_slice]) **2).sort_values().to_frame('r2_score')
display(score)

fig = prob_plots(X_predict, targets['tmax'], predict_df[score.index.values], shape=(3, 3), figsize=(12, 12))
plt.show()