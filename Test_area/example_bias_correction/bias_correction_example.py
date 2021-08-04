import matplotlib.pyplot as plt
import pandas as pd
import sys
import os
from bias_correction import BiasCorrection

os.chdir(sys.path[0])

# obs = pd.read_csv('obs.csv', index_col='time', parse_dates=['time'])
# reanalysis = pd.read_csv('reanalysis.csv', index_col='time', parse_dates=['time'])
# scenario = pd.read_csv('scenario.csv', index_col='time', parse_dates=['time'])

obs = pd.read_csv('/home/phillip/Seafile/Ana-Lena_Phillip/data/matilda/Test_area/example_bias_correction/obs.csv', index_col='time', parse_dates=['time'])
reanalysis = pd.read_csv('/home/phillip/Seafile/Ana-Lena_Phillip/data/matilda/Test_area/example_bias_correction/reanalysis.csv', index_col='time', parse_dates=['time'])
scenario = pd.read_csv('/home/phillip/Seafile/Ana-Lena_Phillip/data/matilda/Test_area/example_bias_correction/scenario.csv', index_col='time', parse_dates=['time'])

## Step 1 - Downscale reananlysis using obs

train_slice = slice('2007-08-10', '2016-01-01')
predict_slice = slice('1982-01-01', '2020-12-31')

x_train = reanalysis[train_slice].squeeze()
y_train = obs[train_slice].squeeze()
x_predict = reanalysis[predict_slice].squeeze()

bc = BiasCorrection(y_train, x_train, x_predict)
t_corr = pd.DataFrame(bc.correct(method='normal_correction'))



# Compare:
first = pd.DataFrame({'obs': y_train, 'mod': x_train, 'result': t_corr[train_slice]['t2m']})
print(first.describe())

first.plot()
plt.title('Daily air temperature in training period')
plt.show()

first.resample('M').agg('mean').plot()
plt.title('Monthly air temperature in training period')
plt.show()




## Step 2 - Downscale scenario data using fitted reanalysis data

train_slice = slice('1982-01-01', '2020-12-31')
predict_slice = slice('1982-01-01', '2100-12-31')

x_train = scenario[train_slice].squeeze()
y_train = t_corr[train_slice].squeeze()
x_predict = scenario[predict_slice].squeeze()

bc_cmip = BiasCorrection(y_train, x_train, x_predict)
t_corr_cmip = pd.DataFrame(bc_cmip.correct(method='normal_mapping'))



# Compare:
second = pd.DataFrame({'obs': y_train, 'mod': x_train, 'result': t_corr_cmip[train_slice]['t2m']})
print(second.describe())

second.plot()
plt.title('Daily air temperature in training period')
plt.show()

second.resample('M').agg('mean').plot()
plt.title('Monthly air temperature in training period')
plt.show()

second.resample('Y').agg('mean').plot()
plt.title('Annual air temperature in training period')
plt.show()
