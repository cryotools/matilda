import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import sys
import socket
from bias_correction import BiasCorrection
import os

warnings.filterwarnings("ignore")  # sklearn
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

aws = pd.read_csv(home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/kyzylsuu/met/obs/' +
'met_data_full_kyzylsuu_2007-2015.csv', parse_dates=["time"], index_col="time")

aws_temp = aws['t2m']            # '2007-08-10' to '2014-12-31'
aws_prec = aws['tp']           # '2007-08-01' to '2016-01-01' with gaps

# Interpolate shorter data gaps and exclude the larger one to avoid NAs:

# temp data gaps: '2010-03-30' to '2010-04-01', '2011-10-12' to '2011-10-31', '2014-05-02' to '2014-05-03',
aws_temp_int1 = aws_temp[slice('2007-08-10', '2011-10-11')]
aws_temp_int2 = aws_temp[slice('2011-11-01', '2016-01-01')]
aws_temp_int1 = aws_temp_int1.interpolate(method='spline', order=2)
aws_temp_int2 = aws_temp_int2.interpolate(method='spline', order=2)
aws_temp_int = pd.concat([aws_temp_int1, aws_temp_int2], axis=0)          # Data gap of 18 days in October 2011


## ERA5L Gridpoint:

era = pd.read_csv(home + '/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/Kysylsuu/' +
't2m_tp_ERA5L_kyzylsuu_42.2_78.2_1982_2019.csv', parse_dates=["time"], index_col="time")

era = era.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})
era_temp = era['t2m']
era_prec = era['tp']

# Include datagap from AWS to align both datasets
era_temp_int1 = era_temp[slice('2007-08-10', '2011-10-11')]
era_temp_int2 = era_temp[slice('2011-11-01', '2016-01-01')]
era_temp_int = pd.concat([era_temp_int1, era_temp_int2], axis=0)      # Data gap of 18 days in October 2011


## CMIP6:

cmip_path = home + '/EBA-CA/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/'
cmip = pd.read_csv(cmip_path + 'CMIP6_mean_42.25-78.25_1980-01-01-2100-12-31.csv',
                   index_col='time', parse_dates=['time'])

scen = ['26', '45', '70', '85']
cmip_edit = {}
for s in scen:
    cmip_scen = cmip.filter(like=s)
    cmip_scen.columns = ['t2m', 'tp']
    cmip_scen = cmip_scen.resample('D').agg({'t2m': 'mean', 'tp': 'sum'})
    cmip_edit[s] = cmip_scen
    # cmip_scen.to_csv(cmip_path + 'CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31' + 'rcp' + s)


##########################
#    Bias adjustment:    #
##########################

## ERA5:

final_train_slice = slice('2007-08-10', '2016-01-01')
final_predict_slice = slice('1982-01-01', '2020-12-31')

# Temperature:
era_corrT = {}

x_train = era_temp_int[final_train_slice].squeeze()
y_train = aws_temp_int[final_train_slice].squeeze()
x_predict = era_temp_int[final_predict_slice].squeeze()
bc_era = BiasCorrection(y_train, x_train, x_predict)
era_corrT = pd.DataFrame(bc_era.correct(method='normal_mapping'))
#    era_corrT.to_csv(era_path + 't2m_CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31_' + 'rcp' + s)


# HIER WEITER MACHEN!!!


# Precipitation:
era_corrP = {}

x_train = era_prec[final_train_slice]['tp'].squeeze()
y_train = aws_prec[final_train_slice]['tp'].squeeze()
x_predict = era_prec[final_predict_slice]['tp'].squeeze()
bc_era = BiasCorrection(y_train, x_train, x_predict)
era_corrP = pd.DataFrame(bc_era.correct(method='gamma_mapping'))
#    era_corrP.to_csv(era_path + 'tp_CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31_' + 'rcp' + s)


# CMIP:

final_train_slice = slice('1982-01-01', '2020-12-31')
final_predict_slice = slice('1982-01-01', '2100-12-31')

# Temperature:
cmip_corrT = {}
for s in scen:
    x_train = cmip_edit[s][final_train_slice]['t2m'].squeeze()
    y_train = era_D[final_train_slice]['t2m'].squeeze()
    x_predict = cmip_edit[s][final_predict_slice]['t2m'].squeeze()
    bc_cmip = BiasCorrection(y_train, x_train, x_predict)
    cmip_corrT[s] = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))
    cmip_corrT[s].to_csv(cmip_path + 't2m_CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31_' + 'rcp' + s)

# Precipitation:
cmip_corrP = {}
for s in scen:
    x_train = cmip_edit[s][final_train_slice]['tp'].squeeze()
    y_train = era_D[final_train_slice]['tp'].squeeze()
    x_predict = cmip_edit[s][final_predict_slice]['tp'].squeeze()
    bc_cmip = BiasCorrection(y_train, x_train, x_predict)
    cmip_corrP[s] = pd.DataFrame(bc_cmip.correct(method='gamma_mapping'))
    cmip_corrP[s].to_csv(cmip_path + 'tp_CMIP6_mean_42.516-79.0167_1982-01-01-2100-12-31_' + 'rcp' + s)