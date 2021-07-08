import numpy as np
import pandas as pd
from math import e
import matplotlib.pyplot as plt

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