import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='darkgrid')
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from skdownscale.pointwise_models import PureAnalog, AnalogRegression
from skdownscale.pointwise_models import BcsdTemperature, BcsdPrecipitation
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
sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/Preprocessing/Downscaling')
from utils import prob_plots


def overview_plot(training, targets, training_var2=None, targets_var2=None, no_var=1, figsize=(8, 6), sharex=True,
                  labelvar1=None, labelvar2=None, label_train="training", label_target="target", **kwargs):
    if no_var == 1:
        fig, axes = plt.subplots(figsize=figsize)
        axes.plot(training, label=label_train)
        axes.plot(targets, label=label_target)
        axes.legend()
        axes.set_ylabel(labelvar1)
    elif no_var == 2:
        fig, axes = plt.subplots(ncols=1, nrows=2, figsize=figsize, sharex=sharex)
        axes[0].plot(training, label=label_train)
        axes[0].plot(targets, label=label_target)
        axes[0].legend()
        axes[0].set_ylabel(labelvar1)

        axes[1].plot(training_var2)
        axes[1].plot(targets_var2)
        _ = axes[1].set_ylabel(labelvar2)
    else:
        print('Too many variables for this function. Please customize the plots yourself.')


def fit_dmodels(x_train, y_train, x_predict, precip=False, **qm_kwargs):

    # Define which models to apply
    if precip:
        models = {
            'GARD: PureAnalog-best-1': PureAnalog(kind='best_analog', n_analogs=1),
            'GARD: PureAnalog-sample-10': PureAnalog(kind='sample_analogs', n_analogs=10),
            'GARD: PureAnalog-weight-10': PureAnalog(kind='weight_analogs', n_analogs=10),
            'GARD: PureAnalog-weight-100': PureAnalog(kind='weight_analogs', n_analogs=100),
            'GARD: PureAnalog-mean-10': PureAnalog(kind='mean_analogs', n_analogs=10),
            # 'GARD: AnalogRegression-100': AnalogRegression(n_analogs=100),
            'GARD: LinearRegression': LinearRegression(),
            'BCSD: BcsdPrecipitation': BcsdPrecipitation(return_anoms=False, **qm_kwargs),  # Only works with decent training period.
            'Sklearn: RandomForestRegressor': RandomForestRegressor(random_state=0)
        }
    else:
        models = {
            'GARD: PureAnalog-best-1': PureAnalog(kind='best_analog', n_analogs=1),
            'GARD: PureAnalog-sample-10': PureAnalog(kind='sample_analogs', n_analogs=10),
            'GARD: PureAnalog-weight-10': PureAnalog(kind='weight_analogs', n_analogs=10),
            'GARD: PureAnalog-weight-100': PureAnalog(kind='weight_analogs', n_analogs=100),
            'GARD: PureAnalog-mean-10': PureAnalog(kind='mean_analogs', n_analogs=10),
            'GARD: AnalogRegression-100': AnalogRegression(n_analogs=100),
            'GARD: LinearRegression': LinearRegression(),
            'BCSD: BcsdTemperature': BcsdTemperature(return_anoms=False),  # Only works with decent training period.
            'Sklearn: RandomForestRegressor': RandomForestRegressor(random_state=0)
        }
    for key, model in models.items():                   # Fit all models
        model.fit(x_train, y_train)
    predict_df = pd.DataFrame(index=x_predict.index)    # store predicted results in this dataframe
    for key, model in models.items():
        predict_df[key] = model.predict(x_predict)

    return {'predictions': predict_df, 'models': models}


def modcomp_plot(targets, x_predict,  predict_df, figsize=(10, 5), xlabel='Date', ylabel=None,
                 savefig=False, fig_name='model_comparison'):

    fig, ax = plt.subplots(figsize=figsize)
    targets.plot(ax=ax, label='target', c='k', lw=1, alpha=0.75, legend=True, zorder=10)
    x_predict.plot(label='original', c='grey', ax=ax, alpha=0.75, legend=True)
    predict_df.plot(ax=ax, lw=0.75)
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    _ = ax.set_ylabel(ylabel)
    _ = ax.set_xlabel(xlabel)
    if savefig:
        fig.savefig(fig_name, bbox_inches='tight')
    else:
        plt.subplots_adjust(right=0.66, bottom=0.15, top=0.95)


def dmod_score(predict_df, targets, y_predict, x_predict, figsize=(12, 12)):

    score = (predict_df.corrwith(y_predict) ** 2).sort_values().to_frame('r2_score')        # calculate r2
    fig = prob_plots(x_predict, targets, predict_df[score.index.values], shape=(3, 3), figsize=figsize)     # QQ-Plots
    return {'R2-scores': score, 'QQ-Matrix': fig}
