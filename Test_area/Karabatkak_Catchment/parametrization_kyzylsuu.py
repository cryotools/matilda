## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
import spotpy
import numpy as np
import socket
import matplotlib.pyplot as plt

#mpl.use('Agg')

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
sys.path.append(home + '/Ana-Lena_Phillip/data/scripts/MATILDA_package_slim')
sys.path.append(home + '/Ana-Lena_Phillip/data/scripts/Test_area')
import mspot_cirrus
from MATILDA_slim import MATILDA

algorithm = 'rope'
rep = 1000

## Setting file paths and parameters                            Ggf. VERALTET!!!
working_directory = home + "/Ana-Lena_Phillip/data/"
input_path = home + "/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/"

data_csv = "ERA5/20210313_42.25-78.25_kyzylsuu_awsq_1982_2019.csv"  # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
# runoff_obs = "obs_kyzylsuu_runoff_1994_1997_zero.csv"  # Daily Runoff Observations in m³/s
runoff_obs = home + "/EBA-CA/Azamat_AvH/workflow/data/Runoff/obs_kyzylsuu_runoff_Hydromet.csv"
#output_path = working_directory + "input_output/output/" + data_csv[4:21]
# output_path = "/home/ana/Desktop/Meeting/kyzylsuu"
output_path = home + "/EBA-CA/Azamat_AvH/workflow/data/MATILDA_runs/" + "Kyzylsuu_testrun"

df = pd.read_csv(input_path + data_csv)
# obs = pd.read_csv(input_path + runoff_obs)
obs = pd.read_csv(runoff_obs)
# obs.set_index('Date', inplace=True)
obs = obs[['Date', 'Qobs']]

plt.ion()

# Basic overview plot
obs_fig = obs.copy()
obs_fig.set_index('Date', inplace=True)
obs_fig.index = pd.to_datetime(obs_fig.index)
plt.figure()
ax = obs_fig.plot(label='Kyzysuu (Hydromet)')
ax.set_ylabel('Discharge [m³/s]')

## Parametrization


if 'node' in host:
    kysyl_par = mspot_cirrus.psample(df=df, obs=obs, rep=rep, dbformat='csv', dbname='kysylsuu' + algorithm + str(rep),
                                     set_up_start='1994-01-01 00:00:00', set_up_end='1995-12-31 23:00:00',
                                     sim_start='1994-01-01 00:00:00', sim_end='1997-12-31 23:00:00', freq="D",
                                     area_cat=315.694,
                                     area_glac=32.51, ele_dat=2550, ele_glac=4074, ele_cat=3225, lr_temp_lo=-0.0065,
                                     lr_temp_up=-0.005,
                                     opt_iter=False, savefig=True, algorithm=algorithm)
else:
    kysyl_par = mspot_cirrus.psample(df=df, obs=obs, rep=3, dbformat=None, dbname='kysylsuu' + algorithm + str(rep),
                                     set_up_start='1994-01-01 00:00:00', set_up_end='1995-12-31 23:00:00',
                                     sim_start='1994-01-01 00:00:00', sim_end='1997-12-31 23:00:00', freq="D",
                                     area_cat=315.694,
                                     area_glac=32.51, ele_dat=2550, ele_glac=4074, ele_cat=3225, lr_temp_lo=-0.0065,
                                     lr_temp_up=-0.005,
                                     opt_iter=False, savefig=False, algorithm=algorithm)

## Filter results
#
# result_path = '/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/kysylsuu_lower_para_sampling_first_cirrus_try'
# results = spotpy.analyser.load_csv_results(result_path)
# best10 = spotpy.analyser.get_posterior(results, percentage=1, maximize=True)      # get best xx%
# # trues = np.where((results['parTT_snow'] < results['parTT_rain']) & (results['parCFMAX_ice'] > results['parCFMAX_snow']))
# trues = results[(results['parTT_snow'] < results['parTT_rain']) & (results['parCFMAX_ice'] > results['parCFMAX_snow'])]
#
# likes = trues['like1']
# maximum = np.nanmax(likes)
# index = np.where(likes == maximum)
#
# best_param = trues[index]
# best_param_values = spotpy.analyser.get_parameters(trues[index])[0]
# par_names = spotpy.analyser.get_parameternames(trues)
# param_zip = zip(par_names, best_param_values)
# best_param = dict(param_zip)



