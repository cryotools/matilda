## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import spotpy  # Load the SPOT package into your working storage
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import nashsutcliffe
from spotpy import analyser  # Load the Plotting extension

home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package_slim')
from MATILDA_slim import MATILDA

## Creating an example file
working_directory = home + "/Seafile/Ana-Lena_Phillip/data/"
input_path_data = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/"
input_path_observations = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"

data_csv = "no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv"  # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
observation_data = "runoff_bashkaindy_04_2019-11_2020_temp_limit.csv"  # Daily Runoff Observations in mm

output_path = working_directory + "input_output/output/" + data_csv[:15]

df = pd.read_csv(input_path_data + data_csv)
obs = pd.read_csv(input_path_observations + observation_data)
obs["Qobs"] = obs["Qobs"] / 86400 * (46.232 * 1000000) / 1000  # Daten in mm, Umrechnung in m3/s

# Creating the MATILDA-class

## Perform parameter sampling (may take a long time depending on # of reps)
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/Test_area')
import mspot_class


def psample(df, obs, rep, dbname='matilda_par_smpl', dbformat=None, obj_func=None, set_up_start=None, set_up_end=None,
            sim_start=None, sim_end=None, freq="D", area_cat=None, area_glac=None,
            ele_dat=None, ele_glac=None, ele_cat=None, interf=4, freqst=2):  # , algorithm='sceua'

    setup = mspot_class.setup(set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start, sim_end=sim_end,
                              freq=freq, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, ele_glac=ele_glac,
                              ele_cat=ele_cat, interf=interf, freqst=freqst)

    spot_setup = setup(df, obs, obj_func)  # Define objective function using obj_func=, otherwise NS-eff is used.
    sampler = spotpy.algorithms.sceua(spot_setup, dbname=dbname, dbformat=dbformat)
    # Change dbformat to None for short tests but to 'csv' or 'sql' to avoid data loss after long calculations

    sampler.sample(rep)  # ideal number of reps = spot_setup.par_iter

    results = sampler.getdata()
    best_param = spotpy.analyser.get_best_parameterset(results)

    bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(results)  # Run with highest NS
    best_model_run = results[bestindex]
    fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = pd.Series(list(list(best_model_run[fields])[0]), index=pd.date_range(sim_start, sim_end))
    # Only necessary because spot_setup.evaluation() has a datetime. Thus both need a datetime.

    fig1 = plt.figure(1, figsize=(9, 5))
    plt.plot(results['like1'])
    plt.ylabel('NS-Eff')
    plt.xlabel('Iteration')

    return [best_param, bestindex, best_model_run, bestobjf, best_simulation, fig1]


best_summary = psample(df=df, obs=obs, rep=3, set_up_start='2018-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                       sim_start='2019-01-01 00:00:00', sim_end='2020-11-01 23:00:00', area_cat=46.232,
                       area_glac=2.566, ele_dat=3864, ele_glac=4042, ele_cat=3360)

best_summary[5].show()

# Weitere Schritte in die Funktion psample
# par.iter irgendwie als Option ermöglichen
# Gesamte Vielfalt der Algorithmen einbauen



# setup = mspot_class.setup(set_up_start = '2018-01-01 00:00:00', set_up_end = '2018-12-31 23:00:00',
#                           sim_start = '2019-01-01 00:00:00', sim_end = '2020-11-01 23:00:00', area_cat = 46.232,
#                           area_glac = 2.566, ele_dat = 3864, ele_glac = 4042, ele_cat = 3360)#, freq = "D")
# spot_setup = setup(df, obs)
# sampler = spotpy.algorithms.sceua(spot_setup, dbname='sceua_matilda', dbformat=None)
# sampler.sample(5)

# Plot results of sampling

# fig = plt.figure(1, figsize=(9, 5))
# plt.plot(results['like1'])
# plt.ylabel('NS-Eff')
# plt.xlabel('Iteration')
# plt.show()

# Find parameter interaction

# spotpy.analyser.plot_parameterInteraction(results)
# posterior = spotpy.analyser.get_posterior(results, percentage=10)
# spotpy.analyser.plot_parameterInteraction(posterior)

# Get best results and plot them

# print(spotpy.analyser.get_best_parameterset(results))
# bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(results)  # Run with highest NS
# best_model_run = results[bestindex]
# fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
# best_simulation = pd.Series(list(list(best_model_run[fields])[0]), index=pd.date_range(sim_start, sim_end))
# Only necessary because spot_setup.evaluation() has a datetime. Thus both need a datetime.

# Plot best run against evaluation series

fig = plt.figure(figsize=(16, 9))
ax = plt.subplot(1, 1, 1)
ax.plot(best_simulation, color='black', linestyle='solid', label='Best objf.=' + str(bestobjf))
ax.plot(spot_setup.evaluation(), 'r.', markersize=3, label='Observation data')
plt.xlabel('Date')
plt.ylabel('Discharge [mm d-1]')
plt.legend(loc='upper right')
plt.show()

# Plot parameter uncertainty

fig = plt.figure(figsize=(16, 9))
ax = plt.subplot(1, 1, 1)
q5, q25, q75, q95 = [], [], [], []
for field in fields:
    q5.append(np.percentile(results[field][-100:-1], 2.5))
    q95.append(np.percentile(results[field][-100:-1], 97.5))
ax.plot(q5, color='dimgrey', linestyle='solid')
ax.plot(q95, color='dimgrey', linestyle='solid')
ax.fill_between(np.arange(0, len(q5), 1), list(q5), list(q95), facecolor='dimgrey', zorder=0,
                linewidth=0, label='parameter uncertainty')
ax.plot(np.array(spot_setup.evaluation()), 'r.',
        label='data')  # Need to remove Timestamp from Evaluation to make compable
ax.set_ylim(0, 100)
ax.set_xlim(0, len(spot_setup.evaluation()))
ax.legend()
plt.show()

## Find best Algorithm

results = []
spot_setup = spot_setup(df, obs)  # Kann man aus irgendeinem Grund nur einmal ausführen.
rep = 50  # ideal number of iterations: spot_setup.par_iter
timeout = 10  # Given in Seconds

parallel = "seq"
dbformat = None  # Change to 'csv' or 'sql' to avoid data loss after long calculations
modelname = 'MATILDA'

sampler = spotpy.algorithms.mc(spot_setup, parallel=parallel, dbname=modelname + '_MC', dbformat=dbformat,
                               sim_timeout=timeout)
sampler.sample(rep)
results.append(sampler.getdata())

sampler = spotpy.algorithms.lhs(spot_setup, parallel=parallel, dbname=modelname + '_LHS', dbformat=dbformat,
                                sim_timeout=timeout)
sampler.sample(rep)
results.append(sampler.getdata())

sampler = spotpy.algorithms.mle(spot_setup, parallel=parallel, dbname=modelname + '_MLE', dbformat=dbformat,
                                sim_timeout=timeout)
sampler.sample(rep)
results.append(sampler.getdata())

sampler = spotpy.algorithms.mcmc(spot_setup, parallel=parallel, dbname=modelname + '_MCMC', dbformat=dbformat,
                                 sim_timeout=timeout)
sampler.sample(rep)
results.append(sampler.getdata())

sampler = spotpy.algorithms.sceua(spot_setup, parallel=parallel, dbname=modelname + '_SCEUA', dbformat=dbformat,
                                  sim_timeout=timeout)
sampler.sample(rep, ngs=4)
results.append(sampler.getdata())

sampler = spotpy.algorithms.sa(spot_setup, parallel=parallel, dbname=modelname + '_SA', dbformat=dbformat,
                               sim_timeout=timeout)
sampler.sample(rep)
results.append(sampler.getdata())

# sampler = spotpy.algorithms.demcz(spot_setup, parallel=parallel, dbname=modelname + '_DEMCz', dbformat=dbformat,
#                                   sim_timeout=timeout)
# sampler.sample(rep, nChains=4)
# results.append(sampler.getdata())

# ROPE works for HBV but stops for MATILDA at repetition 34 or so....

# sampler = spotpy.algorithms.rope(spot_setup, parallel=parallel, dbname=modelname + '_ROPE', dbformat=dbformat,
#                                  sim_timeout=timeout)
# sampler.sample(rep)
# results.append(sampler.getdata())

sampler = spotpy.algorithms.abc(spot_setup, parallel=parallel, dbname=modelname + '_ABC', dbformat=dbformat,
                                sim_timeout=timeout)
sampler.sample(rep)
results.append(sampler.getdata())

sampler = spotpy.algorithms.fscabc(spot_setup, parallel=parallel, dbname=modelname + '_FSABC', dbformat=dbformat,
                                   sim_timeout=timeout)
sampler.sample(rep)
results.append(sampler.getdata())

# sampler = spotpy.algorithms.demcz(spot_setup, parallel=parallel, dbname=modelname + '_DEMCZ', dbformat=dbformat,
#                                   sim_timeout=timeout)
# sampler.sample(rep)
# results.append(sampler.getdata())

sampler = spotpy.algorithms.dream(spot_setup, parallel=parallel, dbname=modelname + '_DREAM', dbformat=dbformat,
                                  sim_timeout=timeout)
sampler.sample(rep)
results.append(sampler.getdata())

algorithms = ['mc', 'lhs', 'mle', 'mcmc', 'sceua', 'sa', 'rope', 'abc', 'fscabc', 'dream']  # 'demcz', , 'demcz'
spotpy.analyser.plot_parametertrace_algorithms(results, algorithms, spot_setup)

## Sensitivity Analysis
spot_setup = spot_setup(df, obs)  # only once

sampler = spotpy.algorithms.fast(spot_setup, dbname='MATILDA_FAST', dbformat=None)
sampler.sample(1000)  # minimum 60 to run through,
# ideal number of iterations: spot_setup.par_iter, immer wieder einzelne Zeilen "out of bounds"
results = sampler.getdata()
analyser.plot_fast_sensitivity(results, number_of_sensitiv_pars=2, fig_name="FAST_sensitivity_MATILDA.png")

SI = spotpy.analyser.get_sensitivity_of_fast(results)  # Sensitivity indexes as dict

## Baustellen:

# TT_snow kann höher sein als TT_rain.
# Die CFMAX-Werte stehen in einem fixen Verhältnis.
# Die Correction-Factors überblenden immer die gesamte Sensitivitäts-Analyse
# Sollte die deltaH-Routine in der class mit eingebaut werden?
