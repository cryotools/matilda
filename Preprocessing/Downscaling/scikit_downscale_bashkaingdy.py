##
import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
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
wd = home + '/Ana-Lena_Phillip/data/scripts/Preprocessing/Downscaling'
import os
os.chdir(wd)
sys.path.append(wd)
import scikit_downscale_matilda as sdsm


# interactive plotting?
plt.ion()

## load my data

dat = pd.read_csv('/home/phillip/Seafile/EBA-CA/Tianshan_data/AWS_atbs/atbs_met-data_2017-2020.csv',
                  parse_dates=['datetime'], index_col='datetime')
# dat = dat.resample('D').mean()

training = dat.drop(columns=['temp_era_fitted', 'temp_minikin'])
targets = dat.drop(columns=['rh', 'ws', 'wd'])          # HIER MIT ERA5L-DATEN FORTFAHREN!!

train_slice = slice('2018-09-07', '2019-03-13')
predict_slice = slice('2019-03-14', '2019-09-13')

## example
dat = pd.read_csv(home + "/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/Bash-Kaindy_preprocessed_forcing_data.csv",
                  parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
# dat = dat.resample('D').mean()

training = dat.drop(columns=['temp_era_fitted', 'temp_minikin'])        # Need to be dataframes
targets = dat.drop(columns=['temp_era_fitted', 'temp_era'])             # Need to be dataframes

train_slice = slice('2018-09-07', '2019-09-13')
predict_slice = slice('2018-09-07', '2019-09-13')
plot_slice = slice('2018-09-07', '2019-09-13')

# extract training / prediction data
x_train = training[train_slice]
y_train = targets[train_slice]
x_predict = training[predict_slice]
y_predict = targets[predict_slice]      # Only for finding the best model.

##
sdsm.overview_plot(training[plot_slice], targets[plot_slice], label='Temperature [°C]')
predict_df = sdsm.fit_dmodels(x_train, y_train, x_predict)
sdsm.modcomp_plot(targets[plot_slice], x_predict[plot_slice], predict_df[plot_slice])
sdsm.dmod_score(predict_df, targets['temp_minikin'], y_predict['temp_minikin'], x_predict)
