import socket
from pathlib import Path
import sys
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
import warnings
import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt
import probscale
from matplotlib.legend import Legend
import seaborn as sns
sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/Preprocessing')
from Downscaling.utils import prob_plots
import warnings


## Function to preprocess AWS-Data from SDSS

# FEHLERHAFT! ÜBERARBEITEN!:
def sdss_open(path, celsius=False, resample=True, tz_localize = True, timezone = 'Asia/Bishkek',
              resample_rate='H', resample_method='mean', time_slice=False, time_start=None, time_end=None):
    aws = pd.read_csv(path)
    aws.columns.values[0] = 'datetime'
    aws.set_index(pd.to_datetime(aws.datetime), inplace=True)
    aws = aws.drop(['datetime'], axis=1)
    if tz_localize:
        aws = aws.tz_localize(timezone)
    if celsius:
        aws.iloc[:, 0] = aws.iloc[:, 0] + 273.15
    if resample and resample_method == 'mean':
        aws = aws.resample(resample_rate).mean()
    elif resample and resample_method == 'sum':
        aws = aws.resample(resample_rate).sum()

    if time_slice and time_start is None and time_end is None:
        print("******************************************************************************************")
        print("WARNING! No time slice defined. Please set valid arguments for time_start and/or time_end.")
        print("******************************************************************************************")
    elif time_slice:
        aws = aws[time_start: time_end]
    return aws

## Function to calculate lapse rate from timeseries

def lapseR(high_values, low_values, alt_high, alt_low, unit='K/m',
           seasonal=False, season=None, summer=[4,5,6,7,8,9,10,11], winter=[12,1,2,3]):
    if seasonal and season == 'summer':
        lapseR = (high_values[high_values.index.month.isin(summer)].mean()
                  - low_values[low_values.index.month.isin(summer)].mean()) / (alt_high - alt_low)
        print('The lapse rate between', alt_low, 'm and', alt_high, 'm in', season, 'is', round(lapseR, 5), unit)
    elif seasonal and season == 'winter':
        lapseR = (high_values[high_values.index.month.isin(winter)].mean()
                  - low_values[low_values.index.month.isin(winter)].mean()) / (alt_high - alt_low)
        print('Lapse rate between', alt_low, 'm and', alt_high, 'm in', season, 'is', round(lapseR, 5), unit)
    else:
        lapseR = (high_values.mean()-low_values.mean()) / (alt_high-alt_low)
        print('The lapse rate', alt_low, 'm and', alt_high, 'm is', round(lapseR, 5), unit)
    return lapseR


## Preprocess data from HOBO Temp/Hum sensors


def hobo_open(path, tz_localize=True, timezone='UTC', time_slice=False, time_start=None, time_end=None,
              resample=False, resample_rate='H', resample_method='mean'):
    hobo = pd.read_csv(path, usecols=(["Date Time - UTC", "Temp, (*C)", "RH, (%)", "DewPt, (*C)"]))
    hobo.columns = ['datetime', 'temp', 'rh', 'dt']
    if hobo.temp.dtype is not np.dtype('float64'):
        hobo = hobo[hobo != ' '].dropna()
    hobo.set_index(pd.to_datetime(hobo.datetime), inplace=True)
    hobo = hobo.drop(['datetime'], axis=1)
    hobo = hobo.apply(pd.to_numeric, errors='coerce')
    hobo.iloc[:, [0, 2]] = hobo.iloc[:, [0, 2]] + 273.15
    if tz_localize:
        hobo = hobo.tz_localize(timezone)
    if resample and resample_method == 'mean':
        hobo = hobo.resample(resample_rate).mean()
    elif resample and resample_method == 'sum':
        hobo = hobo.resample(resample_rate).sum()
    if time_slice and time_start is None and time_end is None:
        print("******************************************************************************************")
        print("WARNING! No time slice defined. Please set valid arguments for time_start and/or time_end.")
        print("******************************************************************************************")
    elif time_slice:
        hobo = hobo[time_start: time_end]
    return hobo


## Transfer function to correct tipping bucket data for solid precipitation undercatch (Kochendorfer et.al. 2020)


def pce_correct(U, t2m, tp, measurement_h=2):
    """Transfer function to correct tipping bucket data for solid precipitation undercatch.
    Divides passed precipitation data by a wind dependent catch efficiency.
    Refers to EQ2 & Table 3 from Kochendorfer et.al. 2020. """

    a_gh_mix = 0.726    # gh = at gauging bucket height
    b_gh_mix = 0.0495   # mix = mixed precip (2° ≥ Tair ≥ −2°C)
    a_gh_solid = 0.701  # solid = solid precip (Tair < −2°C)
    b_gh_solid = 0.227
    U_thresh_gh = 6.1   # maximum wind speed

    a_10m_mix = 0.722   # 10m = at 10m height
    b_10m_mix = 0.0354
    a_10m_solid = 0.7116
    b_10m_solid = 0.1925
    U_thresh_10m = 8

    def cond_solid(U_thresh):
        return (U <= U_thresh) & (t2m <= 271.15)

    def cond_mix(U_thresh):
        return (U <= U_thresh) & (275.15 >= t2m) & (t2m >= 271.15)

    if measurement_h < 7:
        cond_solid = cond_solid(U_thresh_gh)
        cond_mix = cond_mix(U_thresh_gh)

        tp[cond_solid] = tp[cond_solid] / ((a_gh_solid) * e ** (-b_gh_solid * U[cond_solid]))
        tp[cond_mix] = tp[cond_mix] / ((a_gh_mix) * e ** (-b_gh_mix * U[cond_mix]))

    else:
        cond_solid = cond_solid(U_thresh_10m)
        cond_mix = cond_mix(U_thresh_10m)

        tp[cond_solid] = tp[cond_solid] / ((a_10m_solid) * e ** (-b_10m_solid * U[cond_solid]))
        tp[cond_mix] = tp[cond_mix] / ((a_10m_mix) * e ** (-b_10m_mix * U[cond_mix]))

    return tp

##


def trendline(Y, **kwargs):
    """Fits a linear trend line through a passed timeseries
    and adds it to a plot."""

    X = range(len(Y.index))
    z = np.polyfit(X, Y, 1)
    p = np.poly1d(z)
    x = pd.DataFrame(p(X), index=Y.index)
    plt.plot(x, "r--", **kwargs)

##


def consec_days(s, thresh, Nmin):
    """Finds periods of Nmin consecutive days below a threshold."""

    m = np.logical_and.reduce([s.shift(-i).le(thresh) for i in range(Nmin)])
    if Nmin > 1:
        m = pd.Series(m, index=s.index).replace({False: np.NaN}).ffill(limit=Nmin-1).fillna(False)
    else:
        m = pd.Series(m, index=s.index)

    # Form consecutive groups
    gps = m.ne(m.shift(1)).cumsum().where(m)

    return gps

##


def daily_annual_T(x, t):
    x = x[t][['t2m']]
    x["month"] = x.index.month
    x["day"] = x.index.day
    day1 = x.index[0]
    x = x.groupby(["month", "day"]).mean()
    date = pd.date_range(day1, freq='D', periods=len(x)).strftime('%Y-%m-%d')
    x = x.set_index(pd.to_datetime(date))
    return x


##

def prob_plot(original, target, corrected, title=None, ylabel="Temperature [C]", **kwargs):
    fig, ax = plt.subplots(sharex=True, sharey=True)
    scatter_kws = dict(label="", marker=None, linestyle="-")
    common_opts = dict(plottype="qq", problabel="", datalabel="", **kwargs)

    scatter_kws["label"] = "original"
    fig = probscale.probplot(original, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "target"
    fig = probscale.probplot(target, ax=ax, scatter_kws=scatter_kws, **common_opts)

    scatter_kws["label"] = "corrected"
    fig = probscale.probplot(corrected, ax=ax, scatter_kws=scatter_kws, **common_opts)
    ax.set_title(title)
    ax.legend()

    ax.set_xlabel("Standard Normal Quantiles")
    ax.set_ylabel(ylabel)
    fig.tight_layout()

##

def dmod_score(predict_df, targets, x_predict, figsize=(10, 10), shape=(3, 3), **kwargs):
    score = (predict_df.corrwith(targets) ** 2).sort_values().to_frame('r2_score')  # calculate r2
    if predict_df.shape[1] == 1:
        fig = prob_plot(predict_df, targets, x_predict, figsize=figsize, **kwargs)
    else:
        fig = prob_plots(x_predict, targets, predict_df[score.index.values], shape=shape,
                         figsize=figsize)  # QQ-Plots
    return {'R2-score(s)': score, 'QQ-Matrix': fig}


##


def df2long(df, intv_sum='M', intv_mean='Y', rm_col = True, precip=False):       # Convert dataframes to long format for use in seaborn-lineplots.
    if precip:
        if rm_col:
            df = df.iloc[:, :-1].resample(intv_sum).sum().resample(intv_mean).mean()   # Exclude 'mean' column
        else:
            df = df.resample(intv_sum).sum().resample(intv_mean).mean()
        df = df.reset_index()
        df = df.melt('time', var_name='model', value_name='tp')
    else:
        if rm_col:
            df = df.iloc[:, :-1].resample(intv_mean).mean()   # Exclude 'mean' column
        else:
            df = df.resample(intv_mean).mean()
        df = df.reset_index()
        df = df.melt('time', var_name='model', value_name='t2m')
    return df

##


def cmip_plot_ensemble(cmip, era, precip=False, intv_sum='M', intv_mean='Y', figsize=(10, 6), show=True):
    warnings.filterwarnings(action='ignore')
    figure, axis = plt.subplots(figsize=figsize)
    if precip:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_sum=intv_sum, intv_mean=intv_mean, precip=True)
            sns.lineplot(data=df, x='time', y='tp')
        axis.set(xlabel='Year', ylabel='Mean Precipitation [mm]')
        if intv_sum=='M':
            figure.suptitle('Mean Monthly Precipitation [mm]', fontweight='bold')
        elif intv_sum=='Y':
            figure.suptitle('Mean Annual Precipitation [mm]', fontweight='bold')
        era_plot = axis.plot(era.resample(intv_sum).sum().resample(intv_mean).mean(), linewidth=1.5, c='black',
                             label='adjusted ERA5', linestyle='dashed')
    else:
        for i in cmip.keys():
            df = df2long(cmip[i], intv_mean=intv_mean)
            sns.lineplot(data=df, x='time', y='t2m')
        axis.set(xlabel='Year', ylabel='Mean Air Temperature [K]')
        if intv_mean=='10Y':
            figure.suptitle('Mean 10y Air Temperature [K]', fontweight='bold')
        elif intv_mean == 'Y':
            figure.suptitle('Mean Annual Air Temperature [K]', fontweight='bold')
        elif intv_mean == 'M':
            figure.suptitle('Mean Monthly Air Temperature [K]', fontweight='bold')
        era_plot = axis.plot(era.resample(intv_mean).mean(), linewidth=1.5, c='black',
                         label='adjusted ERA5', linestyle='dashed')
    # axis.legend(cmip.keys(), loc='upper left', frameon=False)  # First legend (SSPs)
    axis.legend(['SSP1', '_ci1', 'SSP2', '_ci2', 'SSP3', '_ci3', 'SSP5'], loc='upper left',
                frameon=False)  # First legend --> Workaround as new seaborn version listed CIs in legend
    leg = Legend(axis, era_plot, ['adjusted ERA5L'], bbox_to_anchor=[0, 0.75], loc='center left',
                 frameon=False)  # Second legend (ERA5)
    axis.add_artist(leg)
    plt.grid()
    if show: plt.show()
    warnings.filterwarnings(action='always')

##


def load_cmip(folder, filename):
    scen = ['ssp1', 'ssp2', 'ssp3', 'ssp5']
    cmip = {}
    for s in scen:
        cmip_corr = pd.read_csv(folder + filename + s + '.csv', index_col='time', parse_dates=['time'])
        cmip[s] = cmip_corr
    return cmip

##

def cmip2df(temp, prec, scen, col):
    df = pd.DataFrame({'T2': temp[scen][col], 'RRR': prec[scen][col]}).reset_index()
    df.columns = ['TIMESTAMP', 'T2', 'RRR']
    return df
