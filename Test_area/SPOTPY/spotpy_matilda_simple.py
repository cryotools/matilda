## import of necessary packages
from pathlib import Path
import sys
#from matilda_sample import df, obs, set_up_start, set_up_end, sim_start, sim_end, freq, area_cat, area_glac, ele_dat, ele_glac,ele_cat
import spotpy
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import nashsutcliffe
import pandas as pd
## import of necessary packages
import pandas as pd
from pathlib import Path
import sys
import spotpy
import numpy as np
import socket

host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'

sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/MATILDA/MATILDA_slim')
sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/Test_area/SPOTPY')
import mspot_cirrus
from MATILDA_slim import MATILDA

##
working_directory = home + "/Ana-Lena_Phillip/data/"
input_path = home + "/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/"

data_csv = "obs_20210313_kyzylsuu_awsq_1982_2019.csv"  # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
runoff_obs = "obs_kashkator_runoff_2017_2018.csv"  # Daily Runoff Observations in m³/s
output_path = working_directory + "input_output/output/" + data_csv[4:21]

df = pd.read_csv(input_path + data_csv)
obs = pd.read_csv(input_path + runoff_obs)
# obs["Qobs"] = obs["Qobs"] / 86400*(46.232*1000000)/1000

set_up_start='2017-01-01 00:00:00'
set_up_end='2018-12-31 23:00:00'
sim_start='2017-01-01 00:00:00'
sim_end='2018-11-01 23:00:00'
freq="D"
area_cat=7.53
area_glac=2.95
ele_dat=2550
ele_glac=3957
ele_cat=3830

##
lr_temp_lo=-0.008
lr_prec_lo=-0.0005
BETA_lo=1
CET_lo=0
FC_lo=50
K0_lo=0.01
K1_lo=0.01
K2_lo=0.001
LP_lo=0.3
MAXBAS_lo=2
PERC_lo=0
UZL_lo=0
PCORR_lo=0.5
TT_snow_lo=-1.5
TT_rain_lo=-1.5
CFMAX_snow_lo=1
CFMAX_ice_lo=1
SFCF_lo=0.4
CFR_snow_lo=0
CFR_ice_lo=0
CWH_lo=0
lr_temp_up=-0.004
lr_prec_up=0
BETA_up=6
CET_up=0.3
FC_up=500
K0_up=0.4
K1_up=0.4
K2_up=0.15
LP_up=1
MAXBAS_up=7
PERC_up=3
UZL_up=500
PCORR_up=1.5
TT_snow_up=2.5
TT_rain_up=2.5
CFMAX_snow_up=10
CFMAX_ice_up=10
SFCF_up=1
CFR_snow_up=0.1
CFR_ice_up=0.1
CWH_up=0.2
##

class spot_setup:
    # defining all parameters and the distribution
    lr_temp, lr_prec, BETA, CET, FC, K0, K1, K2, LP, MAXBAS, PERC, UZL, PCORR, \
    TT_snow, TT_rain, CFMAX_snow, CFMAX_ice, SFCF, CFR_snow, CFR_ice, CWH = [
        Uniform(low=lr_temp_lo, high=lr_temp_up),  # lr_temp
        Uniform(low=lr_prec_lo, high=lr_prec_up),  # lr_prec
        Uniform(low=BETA_lo, high=BETA_up),  # BETA
        Uniform(low=CET_lo, high=CET_up),  # CET
        Uniform(low=FC_lo, high=FC_up),  # FC
        Uniform(low=K0_lo, high=K0_up),  # K0
        Uniform(low=K1_lo, high=K1_up),  # K1
        Uniform(low=K2_lo, high=K2_up),  # K2
        Uniform(low=LP_lo, high=LP_up),  # LP
        Uniform(low=MAXBAS_lo, high=MAXBAS_up),  # MAXBAS
        Uniform(low=PERC_lo, high=PERC_up),  # PERC
        Uniform(low=UZL_lo, high=UZL_up),  # UZL
        Uniform(low=PCORR_lo, high=PCORR_up),  # PCORR
        Uniform(low=TT_snow_lo, high=TT_snow_up),  # TT_snow
        Uniform(low=TT_rain_lo, high=TT_rain_up),  # TT_rain
        Uniform(low=CFMAX_snow_lo, high=CFMAX_snow_up),
        # CFMAX_snow     # CFMAX_ice eigentlich immer doppelt so hoch wie CFMAX_snow! Wie machen?
        Uniform(low=CFMAX_ice_lo, high=CFMAX_ice_up),  # CFMAX_ice
        Uniform(low=SFCF_lo, high=SFCF_up),  # SFCF
        Uniform(low=CFR_snow_lo, high=CFR_snow_up),  # CFR_snow
        Uniform(low=CFR_ice_lo, high=CFR_ice_up),  # CFR_ice
        Uniform(low=CWH_lo, high=CWH_up),  # CWH
    ]

    def __init__(self, df, obs, obj_func=None):
        self.obj_func = obj_func
        self.Input = df
        self.obs = obs

    def simulation(self, x):
        sim = MATILDA.MATILDA_simulation(self.Input, obs=self.obs, lr_temp=x.lr_temp, lr_prec=x.lr_prec,
                                             BETA=x.BETA,
                                             CET=x.CET, FC=x.FC, K0=x.K0, K1=x.K1, K2=x.K2, LP=x.LP, MAXBAS=x.MAXBAS,
                                             PERC=x.PERC, UZL=x.UZL, PCORR=x.PCORR, TT_snow=x.TT_snow,
                                             TT_rain=x.TT_rain,
                                             CFMAX_snow=x.CFMAX_snow, CFMAX_ice=x.CFMAX_ice, SFCF=x.SFCF,
                                             CFR_snow=x.CFR_snow, CFR_ice=x.CFR_ice, CWH=x.CWH,
                                             output=None, set_up_start=set_up_start, set_up_end=set_up_end,
                                             sim_start=sim_start, sim_end=sim_end, freq=freq,
                                             area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat,
                                             ele_glac=ele_glac, ele_cat=ele_cat, plots=False)

        # return sim[366:]  # excludes the first year as a spinup period
        return sim[0].Q_Total

    def evaluation(self):
        obs_preproc = self.obs.copy()
        obs_preproc.set_index('Date', inplace=True)
        obs_preproc.index = pd.to_datetime(obs_preproc.index)
        obs_preproc = obs_preproc[sim_start:sim_end]
        # Changing the input unit from m³/s to mm.
        obs_preproc["Qobs"] = obs_preproc["Qobs"] * 86400 / (area_cat * 1000000) * 1000
        obs_preproc = obs_preproc.resample(freq).sum()
        # expanding the observation period to the length of the simulation, filling the NAs with 0
        idx = pd.date_range(start=sim_start, end=sim_end, freq=freq, name=obs_preproc.index.name)
        obs_preproc = obs_preproc.reindex(idx)
        obs_preproc = obs_preproc.fillna(0)
        return obs_preproc.Qobs

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = nashsutcliffe(evaluation, simulation)  # In den Beispielen ist hier ein Minus davor?!
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like

##
rep = 100
spot_setup = spot_setup(df, obs)
sampler = spotpy.algorithms.mc(spot_setup, dbname='20210421_mpi_mc_hbv', dbformat=None, parallel='mpi')
sampler.sample(rep)

# Mit dieser Variante funktioniert MPI. Welche Funktion verhindert es bei psample?
# Hiddenprints? Yesno? psample Funktion? par_iter?
# console: mpirun -c 2 python3 /home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/spotpy_matilda_simple.py
