## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
import spotpy  # Load the SPOT package into your working storage
import numpy as np
from spotpy import analyser  # Load the Plotting extension
home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/matilda/Test_area/SPOTPY')
import mspot

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



## Perform parameter sampling (may take a long time depending on # of reps)

best_summary = mspot.psample(df=df, obs=obs, rep=3, set_up_start='2018-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
                       sim_start='2019-01-01 00:00:00', sim_end='2020-11-01 23:00:00', area_cat=46.232,
                       area_glac=2.566, ele_dat=3864, ele_glac=4035, ele_cat=3485)

best_summary['par_uncertain_plot'].show()

best_summary['best_param']


# Weitere Schritte in die Funktion psample
# Gesamte Vielfalt der Algorithmen einbauen
# CSVs müssen abgespeichert werden (option)
# Analyse und plotting in Funktion packen, um es auch einfach aus csv reproduzieren zu können
# Routine auf mehreren Kernen laufbar machen.
# Routine auf Cirrus ermöglichen.



# Find parameter interaction

# spotpy.analyser.plot_parameterInteraction(results)
# posterior = spotpy.analyser.get_posterior(results, percentage=10)
# spotpy.analyser.plot_parameterInteraction(posterior)






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


## Analyse MSpot-results from csv
wd = '/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/'
result_path = wd + 'kysylsuusa'
results = spotpy.analyser.load_csv_results(result_path)
# best10 = spotpy.analyser.get_posterior(results, percentage=1, maximize=True)      # get best xx%
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

best_param_df = pd.DataFrame(best_param, index=[0])
best_param_df.to_csv(wd + 'best_param_sa_0,7607.csv')

def load_psample(path, max=True, cond1={'par1': 'parTT_rain', 'operator': 'lessthan', 'par2': 'parTT_snow'}):
    results = spotpy.analyser.load_csv_results(path)
    trues = results[(results['parTT_snow'] < results['parTT_rain']) & (results['parCFMAX_ice'] > results['parCFMAX_snow'])]

    trues = results[(results[cond1.get()])]


    likes = trues['like1']
    if max:
        obj_val = np.nanmax(likes)
    else:
        obj_val = np.nanmin(likes)

    index = np.where(likes == obj_val)
    best_param = trues[index]
    best_param_values = spotpy.analyser.get_parameters(trues[index])[0]
    par_names = spotpy.analyser.get_parameternames(trues)
    param_zip = zip(par_names, best_param_values)
    best_param = dict(param_zip)

    return [best_param, obj_val]

def filt(left, operator, right):
    return operator(left, right)

def lessthan(left, right):
    return filt(left, (lambda a, b: a < b), right)

def greaterthan(left, right):
    return filt(left, (lambda a, b: a > b), right)

cond1={'par1': 'parTT_rain', 'operator': '<', 'par2': 'parTT_snow'}
cond1.get('par1')

if cond1.get('operator') == '<':
    zero = results[lessthan(results[cond1.get('par1')], results[cond1.get('par2')])]
elif cond1.get('operator') == '>':
    zero = results[greaterthan(results[cond1.get('par1')], results[cond1.get('par2')])]


one = results[lessthan(results['parTT_snow'], results['parTT_rain'])]

two = results[filt(results['parTT_snow'],(lambda a, b: a<b), results['parTT_rain'])]

three = results[results['parTT_snow'] < results['parTT_rain']]

## Baustellen:

# TT_snow kann höher sein als TT_rain.
# Die CFMAX-Werte stehen in einem fixen Verhältnis.
# Die Correction-Factors überblenden immer die gesamte Sensitivitäts-Analyse
# Sollte die deltaH-Routine in der class mit eingebaut werden?

##
# karab:
wd = '/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/'
result_path = wd + 'karabatkak_upper_para_sampling'
results = spotpy.analyser.load_csv_results(result_path)
trues = results[(results['parTT_snow'] < results['parTT_rain']) & (results['parCFMAX_ice'] > results['parCFMAX_snow'])]

likes = trues['like1']
maximum = np.nanmax(likes)
index = np.where(likes == maximum)

best_param = trues[index]
best_param_values = spotpy.analyser.get_parameters(trues[index])[0]
par_names = spotpy.analyser.get_parameternames(trues)
param_zip = zip(par_names, best_param_values)
best_param = dict(param_zip)

best_param_df = pd.DataFrame(best_param, index=[0])
best_param_df.to_csv(wd + 'best_param_karab_sceua_0,843.csv')

# compare:
kysyl1 = pd.read_csv(wd + 'best_param_rope_0,7676.csv').transpose()
kysyl2 = pd.read_csv(wd + 'best_param_sa_0,7607.csv').transpose()
karab = pd.read_csv(wd + 'best_param_karab_sceua_0,843.csv').transpose()

kysyl1.reset_index(level=0, inplace=True)
kysyl2.reset_index(level=0, inplace=True)
karab.reset_index(level=0, inplace=True)



merge = pd.merge(kysyl1,kysyl2, on='index')
comp = pd.merge(merge, karab, on='index')
pd.DataFrame(kysyl1, kysyl2)
