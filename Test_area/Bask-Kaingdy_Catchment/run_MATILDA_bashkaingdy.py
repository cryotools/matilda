# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
import spotpy
import numpy as np
import socket
import matplotlib as mpl
from MATILDA_slim import MATILDA

mpl.use('Agg')

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
import mspot


# AUF BASKKAINGDY ANPASSEN!!!


## Setting file paths and parameters
working_directory = home + "/Ana-Lena_Phillip/data/"
input_path = home + "/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/"

data_csv = "obs_20210313_kyzylsuu_awsq_1982_2019.csv"  # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
runoff_obs = "obs_kashkator_runoff_2017_2018.csv"  # Daily Runoff Observations in m³/s
output_path = working_directory + "input_output/output/" + data_csv[4:21]
output_path = "/home/ana/Desktop/Meeting/kashkator_"


df = pd.read_csv(input_path + data_csv)
obs = pd.read_csv(input_path + runoff_obs)
# obs["Qobs"] = obs["Qobs"] / 86400*(46.232*1000000)/1000


# set_up_start='2017-01-01 00:00:00'
# set_up_end='2018-12-31 23:00:00'
# sim_start='2017-01-01 00:00:00'
# sim_end='2018-11-01 23:00:00'
# freq="D"
# area_cat=7.53
# area_glac=2.95
# ele_dat=2550
# ele_glac=3957
# ele_cat=3830

## Parametrization

if 'node' in host:
    karab_par = mspot_cirrus.psample(df=df, obs=obs, rep=3, dbformat='csv', dbname='karabatkak_upper_para_sampling',
                                     set_up_start='2017-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                                     sim_start='2017-01-01 00:00:00', sim_end='2018-11-01 23:00:00', freq="D",
                                     area_cat=7.53, area_glac=2.95, ele_dat=2550, ele_glac=3957, ele_cat=3830,
                                     lr_temp_lo=-0.0065, lr_temp_up=-0.005,
                                     opt_iter=True, savefig=True, algorithm='mcmc')
else:
    karab_par = mspot.psample(df=df, obs=obs, rep=3, dbformat=None, dbname='karabatkak_upper_para_sampling',
                                 set_up_start='2017-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                                 sim_start='2017-01-01 00:00:00', sim_end='2018-11-01 23:00:00', freq="D",
                                 area_cat=7.53, area_glac=2.95, ele_dat=2550, ele_glac=3957, ele_cat=3830,
                                 lr_temp_lo=-0.0065, lr_temp_up=-0.005,
                                 opt_iter=False, savefig=False, algorithm='mcmc')





# karab_par['sampling_plot'].show()
# karab_par['best_run_plot'].show()
# karab_par['par_uncertain_plot'].show()
#
# karab_par['sampling_plot'].savefig('sampling_plot_karab.png')
# karab_par['best_run_plot'].savefig('best_run_plot_karab.png')
# karab_par['par_uncertain_plot'].savefig('par_uncertain_plot_karab.png')
#
# best_par_karab = pd.DataFrame.from_dict(karab_par['best_param'], orient='index')
# best_par_karab.to_csv('best_param_karab.csv')

##
result_path = '/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/kysylsuurope2000'
results = spotpy.analyser.load_csv_results(result_path)
# best10 = spotpy.analyser.get_posterior(results, percentage=1, maximize=True)  # get best xx%
# trues = np.where((results['parTT_snow'] < results['parTT_rain']) & (results['parCFMAX_ice'] > results['parCFMAX_snow']))
trues = results[(results['parTT_snow'] < results['parTT_rain']) & (results['parCFMAX_ice'] > results['parCFMAX_snow'])]

likes = trues['like1']
maximum = np.nanmax(likes)
index = np.where(likes == maximum)

best_param = trues[index]
best_param_values = spotpy.analyser.get_parameters(trues[index])[0]
par_names = spotpy.analyser.get_parameternames(trues)
param_zip = zip(par_names, best_param_values)
best_param = dict(param_zip)

best_par_karab = pd.DataFrame.from_dict(best_param, orient='index')
best_par_karab.to_csv(working_directory + 'scripts/Test_area/Karabatkak_Catchment/' + 'best_param_kysyl_20210511.csv')


# bestindex, bestobjf = index, trues[index]['like1']  # Run with highest NS
# best_model_run = trues[bestindex]
# fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
# best_simulation = pd.Series(list(list(best_model_run[fields])[0]), index=pd.date_range(sim_start, sim_end))
# # Only necessary because spot_setup.evaluation() has a datetime. Thus both need a datetime.
#
# fig1 = plt.figure(1, figsize=(9, 5))
# plt.plot(trues['like1'])
# plt.ylabel('NS-Eff')
# plt.xlabel('Iteration')
# plt.show()
#
# fig2 = plt.figure(figsize=(16, 9))
# ax = plt.subplot(1, 1, 1)
# ax.plot(best_simulation, color='black', linestyle='solid', label='Best objf.=' + str(bestobjf))
# ax.plot(spot_setup.evaluation(), 'r.', markersize=3, label='Observation data')
# plt.xlabel('Date')
# plt.ylabel('Discharge [mm d-1]')
# plt.legend(loc='upper right')
# plt.show()
#
# fig3 = plt.figure(figsize=(16, 9))
# ax = plt.subplot(1, 1, 1)
# q5, q25, q75, q95 = [], [], [], []
# for field in fields:
#     q5.append(np.percentile(trues[field][-100:-1], 2.5))
#     q95.append(np.percentile(trues[field][-100:-1], 97.5))
# ax.plot(q5, color='dimgrey', linestyle='solid')
# ax.plot(q95, color='dimgrey', linestyle='solid')
# ax.fill_between(np.arange(0, len(q5), 1), list(q5), list(q95), facecolor='dimgrey', zorder=0,
#                 linewidth=0, label='parameter uncertainty')
# ax.plot(np.array(spot_setup.evaluation()), 'r.',
#         label='data')  # Need to remove Timestamp from Evaluation to make comparable
# ax.set_ylim(0, 100)
# ax.set_xlim(0, len(spot_setup.evaluation()))
# ax.legend()
#
#
#
# ##
# parameter = MATILDA.MATILDA_parameter(df, set_up_start='2017-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
#                                       sim_start='2017-01-01 00:00:00', sim_end='2018-12-31 23:00:00', freq="D",
#                                       area_cat=7.53, area_glac=2.95,
#                                       ele_dat=2550, ele_glac=3957, ele_cat=3830, **best_summary['best_param'])
# ## Running MATILDA
parameter = MATILDA.MATILDA_parameter(df, set_up_start='2016-01-01 00:00:00', set_up_end='2016-12-31 23:00:00',
                                      sim_start='2017-01-01 00:00:00', sim_end='2018-12-31 23:00:00', freq="D",
                                      area_cat=7.53, area_glac=2.95,
                                      ele_dat=2550, ele_glac=3957, lr_temp=-0.005936, lr_prec=-0.0002503,
                                      TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4.824, CFMAX_ice=5.574, CFR_snow=0.08765,
                                      CFR_ice=0.01132, BETA=2.03, CET=0.0471, FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277,
                                      LP=0.4917, MAXBAS=2.494, PERC=1.723, UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765)
df_preproc, obs_preproc = MATILDA.MATILDA_preproc(df, parameter, obs=obs)  # Data preprocessing
#
output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, obs_preproc)  # MATILDA model run + downscaling
#
output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
# # Creating plot for the input (meteorological) data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3) and
# # adding them to the output
#
MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path)
#
# ## This function is a standalone function to run the whole MATILDA simulation
# # If output = output_path in function, the output will be saved to a new folder
# output_MATILDA = MATILDA.MATILDA_simulation(df, obs=obs, set_up_start='2017-01-01 00:00:00',
#                                             set_up_end='2018-12-31 23:00:00',
#                                             sim_start='2017-01-01 00:00:00', sim_end='2018-11-01 23:00:00', freq="D",
#                                             area_cat=7.53, area_glac=2.95,
#                                             ele_dat=2550, ele_glac=3957, ele_cat=3830, TT_snow=0, TT_rain=2)
# output_MATILDA[7].show()
#
# output_MATILDA[0].Q_Total
#
# ##
