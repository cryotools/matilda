import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
sns.set(style='darkgrid')
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from skdownscale.pointwise_models import PureAnalog, AnalogRegression
from skdownscale.pointwise_models import BcsdTemperature, BcsdPrecipitation
from utils import prob_plots


def overview_plot(training, targets, no_var=1, figsize=(8, 6), sharex=True, label=None, label_train="training", label_target="target", **kwargs):
    fig, axes = plt.subplots()
    axes.plot(training, label=label_train)
    axes.plot(targets, label=label_target)
    axes.legend()
    axes.set_ylabel(label)


def fit_dmodels(x_train, y_train, x_predict):

    # Define which models to apply
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


