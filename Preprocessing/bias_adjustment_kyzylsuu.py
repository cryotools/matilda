import pandas as pd
from pathlib import Path
import sys
import socket
from bias_correction import BiasCorrection
import copy as cp
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

##########################
#   Data preparation:    #
##########################

## AWS Chong Kyzylsuu:
aws_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/obs/'
aws = pd.read_csv(aws_path + 'met_data_full_kyzylsuu_2007-2015.csv', parse_dates=["time"], index_col="time")

aws_temp = aws['t2m']  # '2007-08-10' to '2014-12-31'
aws_prec = aws['tp']  # '2007-08-01' to '2016-01-01' with gaps
aws_prec.to_csv(aws_path + 'tp_aws_2007-08-01-2016-01-01.csv')

# Interpolate shorter data gaps and exclude the larger one to avoid NAs:

# temp data gaps: '2010-03-30' to '2010-04-01', '2011-10-12' to '2011-10-31', '2014-05-02' to '2014-05-03',
aws_temp_int1 = aws_temp[slice('2007-08-10', '2011-10-11')]
aws_temp_int2 = aws_temp[slice('2011-11-01', '2016-01-01')]
aws_temp_int1 = aws_temp_int1.interpolate(method='spline', order=2)
aws_temp_int2 = aws_temp_int2.interpolate(method='spline', order=2)
aws_temp_int = pd.concat([aws_temp_int1, aws_temp_int2], axis=0)  # Data gap of 18 days in October 2011
# aws_temp_int.to_csv(aws_path + 't2m-with-gap_aws_2007-08-10-2016-01-01.csv')

## ERA5L Gridpoint:

era_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/era5l/'
era = pd.read_csv(era_path + 't2m_tp_ERA5L_kyzylsuu_42.2_78.2_1982_2020.csv',
                  parse_dates=["time"], index_col="time")

era = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})
era_temp = era['t2m']
era_prec = era['tp']
# era_temp.to_csv(era_path + 't2m_era5l_42.516-79.0167_1982-01-01-2020-12-31.csv')
# era_prec.to_csv(era_path + 'tp_era5l_42.516-79.0167_1982-01-01-2020-12-31.csv')

# Include datagap from AWS to align both datasets
era_temp_int1 = era_temp[slice('2007-08-10', '2011-10-11')]
era_temp_int2 = era_temp[slice('2011-11-01', '2016-01-01')]
era_temp_int = pd.concat([era_temp_int1, era_temp_int2], axis=0)  # Data gap of 18 days in October 2011
# era_temp_int.to_csv(era_path + 't2m-with-gap_era5l_42.516-79.0167_2007-08-10-2016-01-01.csv')

## CMIP6:

cmip_path = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/cmip6/'

# Mean:
cmip = pd.read_csv(cmip_path + 'CMIP6_mean_42.25-78.25_1980-01-01-2100-12-31.csv',
                   index_col='time', parse_dates=['time'])
scen1 = ['26', '45', '70', '85']
cmip_mean = {}
for s in scen1:
    cmip_scen = cmip.filter(like=s)
    cmip_scen.columns = ['t2m', 'tp']
    cmip_scen = cmip_scen.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})
    cmip_scen = cmip_scen.interpolate(method='spline', order=2)  # 3 of 7 models don't have values for Feb 29...
    cmip_mean[s] = cmip_scen
    # cmip_scen.to_csv(cmip_path + 'CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31' + 'rcp' + s)

# Individually:
scen2 = ['1_2_6', '2_4_5', '3_7_0', '5_8_5']
cmip_mod_tas = {}
cmip_mod_pr = {}

# ATTENTION!  The interpolation takes a while:

for s in scen2:
    cmip = pd.read_csv(cmip_path + 'ssp' + s + '_42.25-78.25_1980-01-01-2100-12-31.csv',
                       index_col='time', parse_dates=['time'])
    cmip_tas = cmip.filter(like='tas')
    cmip_tas = cmip_tas.iloc[:, :-1]  # Exclude mean column
    cmip_pr = cmip.filter(like='pr')
    cmip_pr = cmip_pr.iloc[:, :-1]
    cmip_tas = cmip_tas.rename(columns=lambda x: str(x)[4:])  # Crop columns names to model name only 
    cmip_pr = cmip_pr.rename(columns=lambda x: str(x)[3:])
    cmip_tas = cmip_tas.resample('D').mean()  # Reset time from 12:00 to 00:00
    cmip_pr = cmip_pr.resample('D').sum()
    # 3 of 7 models don't have values for Feb 29:
    cmip_tas.loc[(cmip_tas.index.month == 2) | (cmip_tas.index.month == 3), ['inm_cm5_0', 'inm_cm4_8', 'noresm2_mm']] \
        = cmip_tas.loc[(cmip_tas.index.month == 2) | (cmip_tas.index.month == 3), ['inm_cm5_0', 'inm_cm4_8',
                                                                                   'noresm2_mm']].interpolate(
        method='spline', order=2)
    cmip_pr.loc[(cmip_pr.index.month == 2) | (cmip_pr.index.month == 3), ['inm_cm5_0', 'inm_cm4_8', 'noresm2_mm']] \
        = cmip_pr.loc[
        (cmip_pr.index.month == 2) | (cmip_pr.index.month == 3), ['inm_cm5_0', 'inm_cm4_8', 'noresm2_mm']].interpolate(
        method='spline', order=2)
    name = 'ssp' + s[:1]
    cmip_mod_tas[name] = cmip_tas[slice('1982-01-01', '2100-12-31')]
    cmip_mod_pr[name] = cmip_pr[slice('1982-01-01', '2100-12-31')]
    # cmip_mod_tas[name].to_csv(cmip_path + 't2m_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
    #                          + name + '.csv')
    # cmip_mod_pr[name].to_csv(cmip_path + 'tp_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
    #                          + name + '.csv')

##########################
#    Bias adjustment:    #
##########################

### ERA5:

final_train_slice = slice('2007-08-10', '2016-01-01')
final_predict_slice = slice('1982-01-01', '2020-12-31')

# Temperature:

x_train = era_temp_int[final_train_slice].squeeze()
y_train = aws_temp_int[final_train_slice].squeeze()
x_predict = era_temp[final_predict_slice].squeeze()
bc_era = BiasCorrection(y_train, x_train, x_predict)
era_corrT = pd.DataFrame(bc_era.correct(method='normal_mapping'))
# era_corrT.to_csv(era_path + 't2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv')


# Precipitation:

final_train_slice = slice('2007-08-01', '2014-12-31')
final_predict_slice = slice('1982-01-01', '2020-12-31')

x_train = era_prec[final_train_slice].squeeze()
y_train = aws_prec[final_train_slice].squeeze()
x_predict = era_prec[final_predict_slice].squeeze()
bc_era = BiasCorrection(y_train, x_train, x_predict)
era_corrP = pd.DataFrame(bc_era.correct(method='normal_mapping'))
era_corrP[era_corrP < 0] = 0  # only needed when using normal mapping for precipitation
era_corrP.to_csv(era_path + 'tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv')


### CMIP:

final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('1980-01-01', '2100-12-31')

## Mean:

# Temperature:
cmip_corrT_mean = {}
for s in scen1:
    x_train = cmip_mean[s][final_train_slice]['t2m'].squeeze()
    y_train = era_corrT[final_train_slice]['t2m'].squeeze()
    x_predict = cmip_mean[s][final_predict_slice]['t2m'].squeeze()
    bc_cmip = BiasCorrection(y_train, x_train, x_predict)
    cmip_corrT_mean[s] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
    # cmip_corrT_mean[s].to_csv(cmip_path + 't2m_CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31_' + 'rcp' + s + '.csv')

# Precipitation:
cmip_corrP_mean = {}
for s in scen1:
    x_train = cmip_mean[s][final_train_slice]['tp'].squeeze()
    y_train = era_corrP[final_train_slice].squeeze()
    x_predict = cmip_mean[s][final_predict_slice]['tp'].squeeze()
    bc_cmip = BiasCorrection(y_train, x_train, x_predict)
    cmip_corrP_mean[s] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
    cmip_corrP_mean[s][cmip_corrP_mean[s] < 0] = 0  # only needed when using normal mapping for precipitation
    # cmip_corrP_mean[s].to_csv(cmip_path + 'tp_CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31_' + 'rcp' + s + '.csv')

## Individually:

# Temperature:
cmip_corrT_mod = cmip_mod_tas.copy()
for s in scen2:
    s = 'ssp' + s[:1]
    for m in cmip_mod_tas[s].columns:
        x_train = cmip_mod_tas[s][m][final_train_slice].squeeze()
        y_train = era_corrT[final_train_slice]['t2m'].squeeze()
        x_predict = cmip_mod_tas[s][m][final_predict_slice].squeeze()
        bc_cmip = BiasCorrection(y_train, x_train, x_predict)
        cmip_corrT_mod[s][m] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
        cmip_corrT_mod[s]['mean'] = cmip_corrT_mod[s].mean(axis=1)
        cmip_corrT_mod[s].to_csv(cmip_path + 't2m_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_'
                                 + s + '.csv')

# Precipitation:
cmip_corrP_mod = cmip_mod_pr.copy()
for s in scen2:
    s = 'ssp' + s[:1]
    for m in cmip_mod_pr[s].columns:
        x_train = cmip_mod_pr[s][m][final_train_slice].squeeze()
        y_train = era_corrP[final_train_slice].squeeze()
        x_predict = cmip_mod_pr[s][m][final_predict_slice].squeeze()
        bc_cmip = BiasCorrection(y_train, x_train, x_predict)
        cmip_corrP_mod[s][m] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
        cmip_corrP_mod[s][m][cmip_corrP_mod[s][m] < 0] = 0          # only needed when using normal mapping for precipitation
        cmip_corrP_mod[s]['mean'] = cmip_corrP_mod[s].mean(axis=1)
        cmip_corrP_mod[s].to_csv(cmip_path + 'tp_CMIP6_all_models_adjusted_42.516-79.0167_1982-01-01-2100-12-31_'
                                 + s + '.csv')



## Debugging "inm" models
#
# import matplotlib.pyplot as plt
# # from plots_and_stats_kyzylsuu import df2long
# import seaborn as sns
#
# def df2long(df, intv_sum='M', intv_mean='Y', rm_col = True, precip=False):       # Convert dataframes to long format for use in seaborn-lineplots.
#     if precip:
#         if rm_col:
#             df = df.iloc[:, :-1].resample(intv_sum).sum().resample(intv_mean).mean()   # Exclude 'mean' column
#         else:
#             df = df.resample(intv_sum).sum().resample(intv_mean).mean()
#         df = df.reset_index()
#         df = df.melt('time', var_name='model', value_name='tp')
#     else:
#         if rm_col:
#             df = df.iloc[:, :-1].resample(intv_mean).mean()   # Exclude 'mean' column
#         else:
#             df = df.resample(intv_mean).mean()
#         df = df.reset_index()
#         df = df.melt('time', var_name='model', value_name='t2m')
#     return df
#
# final_train_slice = slice('1982-01-01', '2020-12-31')
# final_predict_slice = slice('1982-01-01', '2100-12-31')
#
# dat = pd.read_csv(cmip_path + 'tp_CMIP6_all_models_raw_42.516-79.0167_1982-01-01-2100-12-31_'
#                           + 'ssp1' + '.csv', index_col='time', parse_dates=['time'])
# dat = dat.filter(like='inm')
# dat = dat[slice('1982-01-01', '2100-12-31')]
# inm_corr = cp.deepcopy(dat)
# inm_corr = inm_corr[slice('1982-01-01', '2020-12-31')]
#
# x_train = dat['inm_cm4_8'][final_train_slice].squeeze()
# y_train = era_corrP[final_train_slice].squeeze()
# x_predict = dat['inm_cm4_8'][final_predict_slice].squeeze()
# bc_cmip = BiasCorrection(y_train, x_train, x_predict)
# inm_corr['inm_cm4_8'] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
#
# x_train = dat['inm_cm5_0'][final_train_slice].squeeze()
# y_train = era_corrP[final_train_slice].squeeze()
# x_predict = dat['inm_cm5_0'][final_predict_slice].squeeze()
# bc_cmip = BiasCorrection(y_train, x_train, x_predict)
# inm_corr['inm_cm5_0'] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
#
# inm_corr['era5l'] = era_corrP
#
# inm_corr.sum()
# dat.sum()
#
# figure, axis = plt.subplots(2, 1, figsize=(10, 14), tight_layout=True, sharex='col')
# df = df2long(dat, rm_col=False, precip=True, intv_mean='Y', intv_sum='Y')
# sns.violinplot(ax=axis[0], x='tp', y='model', data=df, scale="count", bw=.2)
# axis[0].set(xlabel=None, xlim=(0, 1100))
# df = df2long(inm_corr, rm_col=False, precip=True, intv_mean='Y', intv_sum='Y')
# sns.violinplot(ax=axis[1], x='tp', y='model', data=df, scale="count", bw=.2)
# axis[1].set(xlabel=None, xlim=(0, 1100))
# plt.show()
#
# # - prediction bis 2029 wie erwartet, danach inm_4_8 total off, nach 2067 auch inm_5_0
# # - mit Normalverteilung funktioniert es aber muss bei 0 gekappt werden