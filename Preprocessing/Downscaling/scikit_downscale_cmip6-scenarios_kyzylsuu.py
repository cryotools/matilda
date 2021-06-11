##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
import salem
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
wd = home + '/Ana-Lena_Phillip/data/scripts/Preprocessing'
import os
os.chdir(wd + '/Downscaling')
sys.path.append(wd)
import Downscaling.scikit_downscale_matilda as sds
from Preprocessing_functions import pce_correct
from skdownscale.pointwise_models import BcsdTemperature, BcsdPrecipitation

# interactive plotting?
# plt.ion()

def trendline(Y, **kwargs):
    X = range(len(Y.index))
    z = np.polyfit(X, Y, 1)
    p = np.poly1d(z)
    x = pd.DataFrame(p(X), index=Y.index)
    plt.plot(x, "r--", **kwargs)


##########################
#   Data preparation:    #
##########################

## ERA5 closest gridpoint:

# Import fitted ERA5L data as target data

era_corr = pd.read_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/' +
                'kyzylsuu_ERA5_Land_1982_2020_42.2_78.2_fitted2AWS.csv', parse_dates=['time'], index_col='time')
t_corr = era_corr.drop(columns=['tp'])
t_corr_D = t_corr.resample('D').mean()
p_corr = era_corr.drop(columns=['t2m'])
p_corr_D = p_corr.resample('D').sum()


## CMIP6 data:

cmip = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/' +
                       'CMIP6_mean_42.25-78.25_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip26 = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/' +
                       'ssp1_2_6_42.25-78.25_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip45 = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/' +
                       'ssp2_4_5_42.25-78.25_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip70 = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/' +
                       'ssp3_7_0_42.25-78.25_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip85 = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/' +
                       'ssp5_8_5_42.25-78.25_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])

cmip_temp = cmip.filter(like='temp').resample('D').mean()
cmip_temp = cmip_temp.interpolate(method='spline', order=2)  
cmip_prec = cmip.filter(like='prec').resample('D').sum()


#################################
#    Downscaling temperature    #
#################################

train_slice = slice('1982-01-01', '2020-12-31')
predict_slice = slice('1982-01-01', '2100-12-31')
plot_slice = slice('2010-01-01', '2019-12-31')

t_corr_cmip = pd.DataFrame(index=cmip_temp[predict_slice].index)

for s in list(cmip_temp):

    x_train = pd.DataFrame(cmip_temp[s][train_slice])
    y_train = t_corr_D[train_slice]
    x_predict = pd.DataFrame(cmip_temp[s][predict_slice])
    y_predict = t_corr_D[predict_slice]

    best_mod = BcsdTemperature(return_anoms=False)
    best_mod.fit(x_train, y_train)
    t_corr_cmip[s] = best_mod.predict(x_predict)

freq = 'Y'
fig, ax = plt.subplots(figsize=(12, 8))
# for i in list(t_corr_cmip): trendline(t_corr_cmip[i].resample(freq).mean())
t_corr_cmip.resample(freq).mean().plot(ax=ax, legend=True)
y_predict['t2m'].resample(freq).mean().plot(label='era5l-fitted', ax=ax, legend=True)



#################################
#   Downscaling precipitation   #
#################################

train_slice = slice('1982-01-01', '2020-12-31')
predict_slice = slice('1982-01-01', '2100-12-31')
plot_slice = slice('2010-01-01', '2019-12-31')

p_corr_cmip = pd.DataFrame(index=cmip_prec[predict_slice].index)

for s in list(cmip_prec):

    x_train = pd.DataFrame(cmip_prec[s][train_slice])
    y_train = p_corr_D[train_slice]
    x_predict = pd.DataFrame(cmip_prec[s][predict_slice])
    y_predict = p_corr_D[predict_slice]

    best_mod = BcsdPrecipitation(return_anoms=False)
    best_mod.fit(x_train, y_train)
    p_corr_cmip[s] = best_mod.predict(x_predict)

freq = 'Y'
fig, ax = plt.subplots(figsize=(12, 8))
for i in list(p_corr_cmip): trendline(p_corr_cmip[i].resample(freq).sum())
p_corr_cmip.resample(freq).sum().plot(ax=ax, legend=True, alpha=0.5)
y_predict['tp'].resample(freq).sum().plot(label='era5l-fitted', ax=ax, legend=True)

## Test:

colors = ['darkblue', 'green', 'darkred', 'darkorange']

freq = 'Y'
fig, ax = plt.subplots(figsize=(12, 8))
for i in range(len((p_corr_cmip.columns))):
    col = colors[i]
    trendline(p_corr_cmip.iloc[:, i].resample(freq).sum(), color=col)
    p_corr_cmip.iloc[:, [i]].resample(freq).sum().plot(ax=ax, legend=True, alpha=0.5, color=col)


# Sieht man in den nict downgecaledten Daten dieselben Trends??



##
freq = 'Y'
scen = '45'
time = slice('1982-01-01', '2100-12-31')
fig, ax = plt.subplots(figsize=(12, 8))
trendline(p_corr_cmip['prec_' + scen][time].resample(freq).sum())
p_corr_cmip['prec_' + scen][time].resample(freq).sum().plot(ax=ax, label='cmip-fit_'+scen, legend=True)
trendline(cmip_prec['prec_' + scen][time].resample(freq).sum())
cmip_prec['prec_' + scen][time].resample(freq).sum().plot(label='cmip-orig_'+scen, ax=ax, legend=True)


#######################################
#   Calculates MinMax of all models   #
#######################################
ds = [cmip26, cmip45, cmip70, cmip85]
names = ['cmip26', 'cmip45', 'cmip70', 'cmip85']
cmip_mm = pd.DataFrame(index=cmip26.index)

for d in range(len(ds)):
    cmip_mm['t2m_'+names[d]+'_min'] = ds[d].filter(like='tas_').min(axis=1)
    cmip_mm['t2m_'+names[d]+'_max'] = ds[d].filter(like='tas_').max(axis=1)
    cmip_mm['tp_'+names[d]+'_min'] = ds[d].filter(like='pr_').min(axis=1)
    cmip_mm['tp_'+names[d]+'_max'] = ds[d].filter(like='pr_').max(axis=1)

plt.fill_between(cmip_mm.resample(freq).mean().index, cmip_mm.t2m_cmip26_min.resample(freq).mean(),
                 cmip_mm.t2m_cmip26_max.resample(freq).mean(),
                 color='blue', alpha=0.2)
plt.show()

# Was begrenzt eigentlich die Fächer in klassischen Szenarioplots?

################################
#   Saving final time series   #
################################

cmip_corr = pd.concat([t_corr_cmip, p_corr_cmip], axis=1)

cmip_corr.to_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/' +
                       'CMIP6_mean_42.25-78.25_1980-01-01-2100-12-31_downscaled.csv')




