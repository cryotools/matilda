##
import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns
from bias_correction import BiasCorrection
from pathlib import Path
import sys
import socket
import os
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/matilda/Preprocessing'
os.chdir(wd + '/Downscaling')
sys.path.append(wd)
from Preprocessing_functions import dmod_score, df2long, cmip_plot_ensemble


###############################################################
aws_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/obs/'
era_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/era5l/'
cmip_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/cmip6/'

aws_temp_int = pd.read_csv(aws_path + 't2m-with-gap_aws_2007-08-10-2016-01-01.csv', index_col='time', parse_dates=['time'])
aws_prec = pd.read_csv(aws_path + 'tp_aws_2007-08-01-2016-01-01.csv', index_col='time', parse_dates=['time'])

era_temp = pd.read_csv(era_path + 't2m_era5l_42.516-79.0167_1982-01-01-2020-12-31.csv', index_col='time', parse_dates=['time'])
era_temp_int = pd.read_csv(era_path + 't2m-with-gap_era5l_42.516-79.0167_2007-08-10-2016-01-01.csv', index_col='time', parse_dates=['time'])
era_prec = pd.read_csv(era_path + 'tp_era5l_42.516-79.0167_1982-01-01-2020-12-31.csv', index_col='time', parse_dates=['time'])

era_corrT = pd.read_csv(era_path + 't2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv', index_col='time', parse_dates=['time'])
era_corrP = pd.read_csv(era_path + 'tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv', index_col='time', parse_dates=['time'])

scen = ['ssp1', 'ssp2', 'ssp3', 'ssp5']
status = ['raw', 'adjusted']
cmip_T_mod = {}
cmip_P_mod = {}
cmip_corrT_mod = {}
cmip_corrP_mod = {}
for s in scen:
    cmip_T = pd.read_csv(cmip_path + 't2m_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
                             + s + '.csv', index_col='time', parse_dates=['time'])
    cmip_Tcorr = pd.read_csv(cmip_path + 't2m_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_'
                             + s + '.csv', index_col='time', parse_dates=['time'])
    cmip_T_mod[s] = cmip_T
    cmip_corrT_mod[s] = cmip_Tcorr
for s in scen:
    cmip_P = pd.read_csv(cmip_path + 'tp_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
                         + s + '.csv', index_col='time', parse_dates=['time'])
    cmip_Pcorr = pd.read_csv(cmip_path + 'tp_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_'
                         + s + '.csv', index_col='time', parse_dates=['time'])
    cmip_P_mod[s] = cmip_P
    cmip_corrP_mod[s] = cmip_Pcorr


## Function:

def cmip_plot(ax, df, scenario, precip=False, intv_sum='M', intv_mean='10Y',  era_label=False):
    if not precip:
        ax.plot(df[scenario].resample(intv_mean).mean().iloc[:, :-1], linewidth=0.6)
        ax.plot(df[scenario].resample(intv_mean).mean().iloc[:, -1], linewidth=1, c='black')
        era_plot, = ax.plot(era_corrT.resample(intv_mean).mean(), linewidth=1.5, c='red', label='adjusted ERA5',
                            linestyle='dashed')
    else:
        ax.plot(df[scenario].resample(intv_sum).sum().resample(intv_mean).mean().iloc[:, :-1],
                linewidth=0.6)
        ax.plot(df[scenario].resample(intv_sum).sum().resample(intv_mean).mean().iloc[:, -1],
                linewidth=1, c='black')
        era_plot, = ax.plot(era_corrP.resample(intv_sum).sum().resample(intv_mean).mean(), linewidth=1.5, c='red',
                    label='adjusted ERA5', linestyle='dashed')
    if era_label:
        ax.legend(handles=[era_plot], loc='upper left')
    ax.set_title(scenario)
    ax.grid(True)



## Plots and stats:

# One plot per SSP (4) with 7 models each plus mean and ERA5 (COMPARE BEFORE AND AFTER ADJUSTMENT):

# Temperature:
figure, axis = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="all")
cmip_plot(axis[0, 0], cmip_T_mod, 'ssp1', era_label=True)       # cmip_T_mod
cmip_plot(axis[0, 1], cmip_T_mod, 'ssp2')
cmip_plot(axis[1, 0], cmip_T_mod, 'ssp3')
cmip_plot(axis[1, 1], cmip_T_mod, 'ssp5')
figure.legend(cmip_T_mod['ssp1'].columns, loc='lower right', ncol=4, mode="expand")
figure.tight_layout()
figure.subplots_adjust(bottom=0.13, top=0.9)
figure.suptitle('10y Mean of Air Temperature', fontweight='bold')
plt.show()

figure, axis = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="all")
cmip_plot(axis[0, 0], cmip_corrT_mod, 'ssp1', era_label=True)       # cmip_T_mod
cmip_plot(axis[0, 1], cmip_corrT_mod, 'ssp2')
cmip_plot(axis[1, 0], cmip_corrT_mod, 'ssp3')
cmip_plot(axis[1, 1], cmip_corrT_mod, 'ssp5')
figure.legend(cmip_corrT_mod['ssp1'].columns, loc='lower right', ncol=4, mode="expand")
figure.tight_layout()
figure.subplots_adjust(bottom=0.13, top=0.9)
figure.suptitle('10y Mean of Air Temperature', fontweight='bold')
plt.show()

# Precipitation:
figure, axis = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="all")
cmip_plot(axis[0, 0], cmip_P_mod, 'ssp1', precip=True, era_label=True)         # cmip_P_mod
cmip_plot(axis[0, 1], cmip_P_mod, 'ssp2', precip=True)
cmip_plot(axis[1, 0], cmip_P_mod, 'ssp3', precip=True)
cmip_plot(axis[1, 1], cmip_P_mod, 'ssp5', precip=True)
figure.legend(cmip_P_mod['ssp1'].columns, loc='lower right', ncol=4, mode="expand")
figure.tight_layout()
figure.subplots_adjust(bottom=0.13, top=0.9)
figure.suptitle('10y Mean of Monthly Precipitation', fontweight='bold')
plt.show()


figure, axis = plt.subplots(2, 2, figsize=(8, 8), sharex="col", sharey="all")
cmip_plot(axis[0, 0], cmip_corrP_mod, 'ssp1', precip=True, era_label=True)         # cmip_corrP_mod
cmip_plot(axis[0, 1], cmip_corrP_mod, 'ssp2', precip=True)
cmip_plot(axis[1, 0], cmip_corrP_mod, 'ssp3', precip=True)
cmip_plot(axis[1, 1], cmip_corrP_mod, 'ssp5', precip=True)
figure.legend(cmip_corrP_mod['ssp1'].columns, loc='lower right', ncol=4, mode="expand")
figure.tight_layout()
figure.subplots_adjust(bottom=0.13, top=0.9)
figure.suptitle('10y Mean of Monthly Precipitation', fontweight='bold')
plt.show()


# (LOESS) SMOOTHING/MOVING WINDOW ANSTELLE VON RESAMPLING?


# One plot per SSP (4) with multi-model mean and shaded confidence interval (or standard deviation)

# Temperature:
cmip_plot_ensemble(cmip_corrT_mod, era_corrT, intv_mean='Y')

# Precipitation:
cmip_plot_ensemble(cmip_corrP_mod, era_corrP, precip=True, intv_sum='Y', intv_mean='Y')


## Violinplots for all models per scenario

    # All models (1980-2100)

figure, axis = plt.subplots(4, 1, figsize=(8, 14), tight_layout=True, sharex='col')
for (i, k) in zip(cmip_corrT_mod.keys(), range(0, 4, 1)):
    df = df2long(cmip_corrT_mod[i])
    sns.violinplot(ax=axis[k], x='t2m', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i)
plt.xlabel('Air Temperature [K]')
figure.suptitle('Kernel Density Estimation of Mean Annual Air Temperature', fontweight='bold')
plt.show()

figure, axis = plt.subplots(4, 1, figsize=(8, 14), tight_layout=True, sharex='col')
for (i, k) in zip(cmip_corrP_mod.keys(), range(0, 4, 1)):
    df = df2long(cmip_corrP_mod[i], precip=True, intv_mean='Y', intv_sum='Y')
    sns.violinplot(ax=axis[k], x='tp', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i)
plt.xlabel('Precipitation [mm]')
figure.suptitle('Kernel Density Estimation of Mean Annual Precipitation', fontweight='bold')
plt.show()


    # Training period incl. target data:

df_vioT = cmip_corrT_mod.copy()
for i in df_vioT.keys():
    df_vioT[i] = df_vioT[i][slice('1982-01-01', '2020-12-31')]
    df_vioT[i]['adj. ERA5L'] = era_corrT[slice('1982-01-01', '2020-12-31')]

figure, axis = plt.subplots(4, 1, figsize=(8, 14), tight_layout=True, sharex='col')
for (i, k) in zip(df_vioT.keys(), range(0, 4, 1)):
    df = df2long(df_vioT[i], rm_col=False)
    sns.violinplot(ax=axis[k], x='t2m', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i)
plt.xlabel('Air Temperature [K]')
figure.suptitle('Kernel Density Estimation of Mean Annual Air Temperature (1982-2020)', fontweight='bold')
plt.show()

df_vioP = cmip_corrT_mod.copy()
for i in df_vioP.keys():
    df_vioP[i] = df_vioP[i][slice('1982-01-01', '2020-12-31')]
    df_vioP[i]['adj. ERA5L'] = era_corrT[slice('1982-01-01', '2020-12-31')]
    
figure, axis = plt.subplots(4, 1, figsize=(8, 14), tight_layout=True, sharex='col')
for (i, k) in zip(df_vioP.keys(), range(0, 4, 1)):
    df = df2long(df_vioP[i], rm_col=False, precip=True, intv_mean='Y', intv_sum='Y')
    sns.violinplot(ax=axis[k], x='tp', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i)
plt.xlabel('Precipitation [mm]')
figure.suptitle('Kernel Density Estimation of Annual Precipitation (1982-2020)', fontweight='bold')
plt.show()



    ## For raw AND adjusted data:
    
df_vioTcorr = cmip_corrT_mod.copy()
df_vioT = cmip_T_mod.copy()

df_vioT.values()
for i in df_vioT.keys():
    df_vioT[i] = df_vioT[i][slice('1982-01-01', '2020-12-31')]
    df_vioT[i]['adj. ERA5L'] = era_corrT[slice('1982-01-01', '2020-12-31')]

for i in df_vioTcorr.keys():
    df_vioTcorr[i] = df_vioTcorr[i][slice('1982-01-01', '2020-12-31')]
    df_vioTcorr[i]['adj. ERA5L'] = era_corrT[slice('1982-01-01', '2020-12-31')]
    df_vioTcorr[i].drop('mean', axis=1, inplace=True)


fig = plt.figure(figsize=(20, 20))
outer = fig.add_gridspec(1, 2)

inner = outer[0].subgridspec(4, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(df_vioT.keys(), range(0, 4, 1)):
    df = df2long(df_vioT[i], rm_col=False)
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='t2m', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i, xlim=(267.5, 287.5))
    if k == 0:
        axis[k].set_title('Before Scaled Distribution Mapping')
plt.xlabel('Air Temperature [K]')

inner = outer[1].subgridspec(4, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(df_vioTcorr.keys(), range(0, 4, 1)):
    df = df2long(df_vioTcorr[i], rm_col=False)
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='t2m', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i, xlim=(267.5, 287.5))
    axis[k].get_yaxis().set_visible(False)
    if k == 0:
        axis[k].set_title('After Scaled Distribution Mapping')
plt.xlabel('Air Temperature [K]')

fig.suptitle('Kernel Density Estimation of Mean Annual Air Temperature (1982-2020)', fontweight='bold', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
plt.show()



df_vioPcorr = cmip_corrP_mod.copy()
df_vioP = cmip_P_mod.copy()

df_vioP.values()
for i in df_vioP.keys():
    df_vioP[i] = df_vioP[i][slice('1982-01-01', '2020-12-31')]
    df_vioP[i]['adj. ERA5L'] = era_corrP[slice('1982-01-01', '2020-12-31')]

for i in df_vioPcorr.keys():
    df_vioPcorr[i] = df_vioPcorr[i][slice('1982-01-01', '2020-12-31')]
    df_vioPcorr[i]['adj. ERA5L'] = era_corrP[slice('1982-01-01', '2020-12-31')]
    df_vioPcorr[i].drop('mean', axis=1, inplace=True)

fig = plt.figure(figsize=(20, 20))  # , constrained_layout=True)
outer = fig.add_gridspec(1, 2)

inner = outer[0].subgridspec(4, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(df_vioP.keys(), range(0, 4, 1)):
    df = df2long(df_vioP[i], rm_col=False, precip=True, intv_mean='Y', intv_sum='Y')
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='tp', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i, xlim=(0,1700))
    if k == 0:
        axis[k].set_title('Before Scaled Distribution Mapping')
plt.xlabel('Annual Precipitation [mm]')

inner = outer[1].subgridspec(4, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(df_vioPcorr.keys(), range(0, 4, 1)):
    df = df2long(df_vioPcorr[i], rm_col=False, precip=True, intv_mean='Y', intv_sum='Y')
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='tp', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i, xlim=(0,1700))
    axis[k].get_yaxis().set_visible(False)
    if k == 0:
        axis[k].set_title('After Scaled Distribution Mapping')
plt.xlabel('Annual Precipitation [mm]')

fig.suptitle('Kernel Density Estimation of Annual Precipitation (1982-2020)', fontweight='bold', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
plt.show()


    ## For full scenario:

df_vioTcorr = cmip_corrT_mod.copy()
df_vioT = cmip_T_mod.copy()

for i in df_vioTcorr.keys():
    df_vioTcorr[i].drop('mean', axis=1, inplace=True)

fig = plt.figure(figsize=(20, 20))
outer = fig.add_gridspec(1, 2)

inner = outer[0].subgridspec(4, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(df_vioT.keys(), range(0, 4, 1)):
    df = df2long(df_vioT[i], rm_col=False)
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='t2m', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i, xlim=(267.5, 291.5))
    if k == 0:
        axis[k].set_title('Before Scaled Distribution Mapping')
plt.xlabel('Air Temperature [K]')

inner = outer[1].subgridspec(4, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(df_vioTcorr.keys(), range(0, 4, 1)):
    df = df2long(df_vioTcorr[i], rm_col=False)
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='t2m', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i, xlim=(267.5, 291.5))
    axis[k].get_yaxis().set_visible(False)
    if k == 0:
        axis[k].set_title('After Scaled Distribution Mapping')
plt.xlabel('Air Temperature [K]')

fig.suptitle('Kernel Density Estimation of Mean Annual Air Temperature (1982-2100)', fontweight='bold', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
plt.show()




df_vioPcorr = cmip_corrP_mod.copy()
df_vioP = cmip_P_mod.copy()

for i in df_vioPcorr.keys():
    df_vioPcorr[i].drop('mean', axis=1, inplace=True)


fig = plt.figure(figsize=(20, 20))
outer = fig.add_gridspec(1, 2)

inner = outer[0].subgridspec(4, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(df_vioP.keys(), range(0, 4, 1)):
    df = df2long(df_vioP[i], rm_col=False, precip=True, intv_mean='Y', intv_sum='Y')
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='tp', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i, xlim=(0,2000))
    if k == 0:
        axis[k].set_title('Before Scaled Distribution Mapping')
plt.xlabel('Annual Precipitation [mm]')

inner = outer[1].subgridspec(4, 1)
axis = inner.subplots(sharex='col')
for (i, k) in zip(df_vioPcorr.keys(), range(0, 4, 1)):
    df = df2long(df_vioPcorr[i], rm_col=False, precip=True, intv_mean='Y', intv_sum='Y')
    axis[k].grid()
    sns.violinplot(ax=axis[k], x='tp', y='model', data=df, scale="count", bw=.2)
    axis[k].set(xlabel=None, ylabel=i, xlim=(0,2000))
    axis[k].get_yaxis().set_visible(False)
    if k == 0:
        axis[k].set_title('After Scaled Distribution Mapping')
plt.xlabel('Annual Precipitation [mm]')

fig.suptitle('Kernel Density Estimation of Annual Precipitation (1982-2100)', fontweight='bold', fontsize=20)
fig.tight_layout()
fig.subplots_adjust(top=0.93)
plt.show()



## Stats
stats = {}
stats_corr = {}
for i in scen:
    d = df_vioT[i].describe()
    dc = df_vioTcorr[i].describe()
    stats[i] = d
    stats_corr[i] = dc


## WORK IN PROGRESS

## Quality assessment:

def bc_check(x_train, y_train, x_predict, y_predict, train_slice, predict_slice, precip = False):

    x_train = x_train[train_slice].squeeze()
    y_train = y_train[train_slice].squeeze()
    x_predict = x_predict[predict_slice].squeeze()
    y_predict = y_predict[predict_slice].squeeze()
    bc = BiasCorrection(y_train, x_train, x_predict)
    if not precip:
        sdm = pd.DataFrame(bc.correct(method='normal_mapping'))
    else:
        sdm = pd.DataFrame(bc.correct(method='gamma_mapping'))

    return {'x_train': x_train, 'y_train': y_train, 'x_predict': x_predict, 'y_predict': y_predict, 'sdm': sdm}

def q_assess(precip=False, **bc_check):
    if not precip:
        result = dmod_score(bc_check['sdm'], bc_check['y_predict'], bc_check['x_predict'], ylabel="Temperature [C]")
    else:
        result = dmod_score(bc_check['sdm'], bc_check['y_predict'], bc_check['x_predict'], ylabel="Precipitation [mm]")

    print(result['R2-score(s)'])
    return result

# ERA5:
train_slice = slice('2007-08-10', '2012-12-31')
predict_slice = slice('2013-01-01', '2016-01-01')

    # Temperature:
era_check = bc_check(era_temp_int, aws_temp_int, era_temp, aws_temp_int, train_slice, predict_slice)
q_assess(**era_check)
plt.show()

    # Precipitation:
era_check = bc_check(era_prec, aws_prec, era_prec, aws_prec, train_slice, predict_slice, precip=True)
test = q_assess(**era_check, precip=True)
plt.show()



# CMIP:
train_slice = slice('1982-01-01', '1999-12-31')
predict_slice = slice('2000-01-01', '2020-12-31')

    # Temperature:

cmip_check = bc_check(cmip_corrT_mod['ssp1'].iloc[:,:1], era_corrT[train_slice], cmip_corrT_mod['ssp1'].iloc[:,:1], era_corrT,
         train_slice, predict_slice)
q_assess(**cmip_check)
plt.show()

for m in cmip_corrT_mod['ssp1'].columns:
    cmip_check = bc_check(cmip_corrT_mod['ssp1'][m], era_corrT,
                          cmip_corrT_mod['ssp1'][m], era_corrT,
                          train_slice, predict_slice)


# WIE LOOPE ICH AM SINNVOLLSTEN DURCH ALLE MODELLE UND SZENARIEN? bc_check SPUCKT JEWEILS EIN dict() AUS. q_assess BZW.
# dmod_score SIND ABER NICHT DAFÜR GEBAUT für x_predict AUCH EINEN DATAFRAME ZU BEKOMMEN.


