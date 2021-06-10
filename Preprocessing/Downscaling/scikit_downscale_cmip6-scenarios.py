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


##########################
#   Data preparation:    #
##########################

## ERA5 closest gridpoint - Bash-Kaingdy:

# Import fitted ERA5L data as target data

era_corr = pd.read_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/' +
                'no182_ERA5_Land_1982_2020_41_75.9_fitted2AWS.csv', parse_dates=['time'], index_col='time')
t_corr = era_corr.drop(columns=['tp'])
t_corr_D = t_corr.resample('D').mean()
p_corr = era_corr.drop(columns=['t2m'])
p_corr_D = p_corr.resample('D').sum()


## CMIP6 data:

cmip = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/' +
                       'CMIP6_mean_41-75.9_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip26 = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/' +
                       'ssp1_2_6_41-75.9_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip45 = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/' +
                       'ssp2_4_5_41-75.9_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])
cmip85 = pd.read_csv(home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/' +
                       'ssp5_8_5_41-75.9_1980-01-01-2100-12-31.csv', index_col='time', parse_dates=['time'])

cmip_temp = cmip.filter(like='temp').resample('D').mean()
cmip_temp = cmip_temp.interpolate(method='spline', order=2)  
cmip_prec = cmip.filter(like='prec').resample('D').sum()


#################################
#    Downscaling temperature    #
#################################

train_slice = slice('1982-01-01', '2020-12-31')
predict_slice = slice('2000-01-01', '2100-12-30')
plot_slice = slice('2010-01-01', '2019-12-31')

t_corr_cmip['new'] = cmip_temp['temp_45']

for s in list(cmip_temp):

    x_train = pd.DataFrame(cmip_temp[s][train_slice])
    y_train = t_corr_D[train_slice]
    x_predict = pd.DataFrame(cmip_temp[s][predict_slice])
    y_predict = t_corr_D[predict_slice]

    best_mod = BcsdTemperature(return_anoms=False)
    best_mod.fit(x_train, y_train)
    t_corr_cmip = pd.DataFrame(index=x_predict.index)
    t_corr_cmip[s] = best_mod.predict(x_predict)

# Spalte wird immer wieder überschrieben. aber warum?