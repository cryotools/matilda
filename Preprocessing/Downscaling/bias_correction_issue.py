from pathlib import Path
import sys
import socket
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/matilda/Preprocessing'
import os
os.chdir(wd + '/Downscaling')
sys.path.append(wd)
from forcing_data_preprocessing_kyzylsuu import era_temp_D_int, aws_temp_D_int, era_temp_D, aws_temp_D, era,\
    aws_temp, cmip, aws_prec, era_D
from bias_correction import BiasCorrection



final_train_slice = slice('2007-08-10', '2016-01-01')
final_predict_slice = slice('1982-01-01', '2020-12-31')
x_train = era_temp_D[final_train_slice]['t2m'].squeeze()
y_train = aws_temp_D[final_train_slice]['t2m'].squeeze()
x_predict = era_temp_D[final_predict_slice]['t2m'].squeeze()

bc = BiasCorrection(y_train, x_train, x_predict)
t_corr = pd.DataFrame(bc.correct(method='normal_correction')) # normal_correction refers to QM but SDM can't cope with NAs

final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('1982-01-01', '2100-12-31')

x_train = cmip[final_train_slice]['t2m'].squeeze()
y_train = t_corr[final_train_slice]['t2m'].squeeze()        # add +6.8 and it fits perfectly
x_predict = cmip[final_predict_slice]['t2m'].squeeze()

bc_cmip = BiasCorrection(y_train, x_train, x_predict)
t_corr_cmip = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))

t_corr_cmip.describe()
x_train.describe()
y_train.describe()
x_predict.describe()



t = slice('2007-08-10', '2016-01-01')
freq = 'M'
fig, ax = plt.subplots(figsize=(6, 4))
# aws_temp_D[t]['t2m'].resample(freq).mean().plot(ax=ax, label='obs', legend=True)
era_temp_D[t]['t2m'].resample(freq).mean().plot(ax=ax, label='era5 (mod)', legend=True)
t_corr[t]['t2m'].resample(freq).mean().plot(label='era5_sdm (result)', ax=ax, legend=True)
ax.set_title('Monthly air temperature in training period')

plt.show()

first = pd.DataFrame({'obs': aws_temp_D[t]['t2m'], 'mod': era_temp_D[t]['t2m'], 'result': t_corr[t]['t2m']})
fig = plt.figure(1, figsize=(6, 4))
ax = fig.add_subplot(111)
bp = ax.boxplot(first['obs'])
plt.show()

first.describe()


t = slice('2017-01-01', '2020-12-31')
freq = 'M'
fig, ax = plt.subplots(figsize=(6, 4))
cmip[t]['t2m'].resample(freq).mean().plot(ax=ax, label='cmip6 (mod)', legend=True)
t_corr_cmip[t]['t2m'].resample(freq).mean().plot(ax=ax, label='cmip6_sdm (result)', legend=True)
t_corr[t]['t2m'].resample(freq).mean().plot(label='era5_sdm (obs)', ax=ax, legend=True)
ax.set_title('Monthly air temperature in example period')

plt.show()

second = pd.DataFrame({'obs': t_corr[t]['t2m'], 'mod': cmip[t]['t2m'], 'result': t_corr_cmip[t]['t2m']})
fig = plt.figure(1, figsize=(6, 4))
ax = fig.add_subplot(111)
bp = ax.boxplot(second)
ax.set_xticklabels(['era5_sdm (obs)', 'cmip6 (mod)', 'cmip6_sdm (result)'])
plt.show()

second.describe()