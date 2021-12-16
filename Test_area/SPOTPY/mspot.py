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
import mspot
home = str(Path.home())
sys.path.append(home + '/Ana-Lena_Phillip/data/matilda/MATILDA/MATILDA_slim')
import MATILDA


# Create the MATILDA-SPOTPy-class

class HiddenPrints:
    """Suppress prints when running multiple iterations."""

    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def yesno(question):
    """Simple Yes/No Function."""
    prompt = f'{question} ? (y/n): '
    ans = input(prompt).strip().lower()
    if ans not in ['y', 'n']:
        print(f'{ans} is invalid, please try again...')
        return yesno(question)
    if ans == 'y':
        return True
    return False


def setup(set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D", area_cat=None, area_glac=None,
          ele_dat=None, ele_glac=None, ele_cat=None, lat=None,
          lr_temp_lo=-0.008, lr_prec_lo=-0.0005, BETA_lo=1, CET_lo=0, FC_lo=50, K0_lo=0.01, K1_lo=0.01, K2_lo=0.001,
          LP_lo=0.3, MAXBAS_lo=2, PERC_lo=0, UZL_lo=0, PCORR_lo=0.5, TT_snow_lo=-1.5, TT_rain_lo=-1.5, CFMAX_snow_lo=1,
          CFMAX_ice_lo=1, SFCF_lo=0.4, CFR_snow_lo=0, CFR_ice_lo=0, CWH_lo=0, lr_temp_up=-0.004, lr_prec_up=0,
          BETA_up=6, CET_up=0.3, FC_up=500, K0_up=0.4, K1_up=0.4, K2_up=0.15, LP_up=1, MAXBAS_up=7, PERC_up=3,
          UZL_up=500, PCORR_up=1.5, TT_snow_up=2.5, TT_rain_up=2.5, CFMAX_snow_up=10, CFMAX_ice_up=10, SFCF_up=1,
          CFR_snow_up=0.1, CFR_ice_up=0.1, CWH_up=0.2,
          interf=4, freqst=2):
    class spot_setup:
        # defining all parameters and the distribution
        param = lr_temp, lr_prec, BETA, CET, FC, K0, K1, K2, LP, MAXBAS, PERC, UZL, PCORR, \
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

        # Number of needed parameter iterations for parametrization and sensitivity analysis
        M = interf  # inference factor (default = 4)
        d = freqst  # frequency step (default = 2)
        k = len(param)  # number of parameters

        par_iter = (1 + 4 * M ** 2 * (1 + (k - 2) * d)) * k

        def __init__(self, df, obs, obj_func=None):
            self.obj_func = obj_func
            self.Input = df
            self.obs = obs

        def simulation(self, x):
            with HiddenPrints():
                sim = MATILDA.MATILDA_simulation(self.Input, obs=self.obs, lr_temp=x.lr_temp, lr_prec=x.lr_prec,
                                                 BETA=x.BETA,
                                                 CET=x.CET, FC=x.FC, K0=x.K0, K1=x.K1, K2=x.K2, LP=x.LP,
                                                 MAXBAS=x.MAXBAS,
                                                 PERC=x.PERC, UZL=x.UZL, PCORR=x.PCORR, TT_snow=x.TT_snow,
                                                 TT_rain=x.TT_rain,
                                                 CFMAX_snow=x.CFMAX_snow, CFMAX_ice=x.CFMAX_ice, SFCF=x.SFCF,
                                                 CFR_snow=x.CFR_snow, CFR_ice=x.CFR_ice, CWH=x.CWH,
                                                 output=None, set_up_start=set_up_start, set_up_end=set_up_end,
                                                 sim_start=sim_start, sim_end=sim_end, freq=freq,
                                                 area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat,
                                                 ele_glac=ele_glac, ele_cat=ele_cat, lat=lat, plots=False)

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

    return spot_setup


def psample(df, obs, rep=10, dbname='matilda_par_smpl', dbformat=None, obj_func=None, opt_iter=False, savefig=False,
            set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D", area_cat=None,
            area_glac=None, ele_dat=None, ele_glac=None, ele_cat=None, lat=None, interf=4, freqst=2, parallel=False, ngs=2,
            algorithm='sceua', **kwargs):

    setup = mspot.setup(set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start, sim_end=sim_end,
                        freq=freq, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, ele_glac=ele_glac,
                        ele_cat=ele_cat, lat=lat, interf=interf, freqst=freqst, **kwargs)

    spot_setup = setup(df, obs, obj_func)  # Define objective function using obj_func=, otherwise NS-eff is used.
    alg_selector = {'mc': spotpy.algorithms.mc, 'sceua': spotpy.algorithms.sceua, 'mcmc': spotpy.algorithms.mcmc,
                    'mle': spotpy.algorithms.mle, 'abc': spotpy.algorithms.abc, 'sa': spotpy.algorithms.sa,
                    'dds': spotpy.algorithms.dds, 'demcz': spotpy.algorithms.demcz,
                    'dream': spotpy.algorithms.dream, 'fscabc': spotpy.algorithms.fscabc,
                    'lhs': spotpy.algorithms.lhs, 'padds': spotpy.algorithms.padds,
                    'rope': spotpy.algorithms.rope}

    if parallel:
        sampler = alg_selector[algorithm](spot_setup, dbname=dbname, dbformat=dbformat, parallel='mpi')
        if opt_iter:
            if yesno("\n******** WARNING! Your optimum # of iterations is {0}. "
                     "This may take a long time.\n******** Do you wish to proceed".format(spot_setup.par_iter)):
                sampler.sample(spot_setup.par_iter, ngs=ngs)  # ideal number of reps = spot_setup.par_iter
            else:
                return
        else:
            sampler.sample(rep)
    else:
        sampler = alg_selector[algorithm](spot_setup, dbname=dbname, dbformat=dbformat)
        if opt_iter:
            if yesno("\n******** WARNING! Your optimum # of iterations is {0}. "
                     "This may take a long time.\n******** Do you wish to proceed".format(spot_setup.par_iter)):
                sampler.sample(spot_setup.par_iter)  # ideal number of reps = spot_setup.par_iter
            else:
                return
        else:
            sampler.sample(rep)

    # if parallel:
    #     sampler = spotpy.algorithms.sceua(spot_setup, dbname=dbname, dbformat=dbformat, parallel='mpi')
    #     if opt_iter:
    #         if yesno("\n******** WARNING! Your optimum # of iterations is {0}. "
    #                  "This may take a long time.\n******** Do you wish to proceed".format(spot_setup.par_iter)):
    #             sampler.sample(spot_setup.par_iter, ngs=ngs)  # ideal number of reps = spot_setup.par_iter
    #         else:
    #             return
    #     else:
    #         sampler.sample(rep)
    # else:
    #     sampler = spotpy.algorithms.sceua(spot_setup, dbname=dbname, dbformat=dbformat)
    #     if opt_iter:
    #         if yesno("\n******** WARNING! Your optimum # of iterations is {0}. "
    #                  "This may take a long time.\n******** Do you wish to proceed".format(spot_setup.par_iter)):
    #             sampler.sample(spot_setup.par_iter)  # ideal number of reps = spot_setup.par_iter
    #         else:
    #             return
    #     else:
    #         sampler.sample(rep)

    # Change dbformat to None for short tests but to 'csv' or 'sql' to avoid data loss in case off long calculations.

    results = sampler.getdata()
    best_param = spotpy.analyser.get_best_parameterset(results)
    par_names = spotpy.analyser.get_parameternames(best_param)
    param_zip = zip(par_names, best_param[0])
    best_param = dict(param_zip)

    bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(results)  # Run with highest NS
    best_model_run = results[bestindex]
    fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
    best_simulation = pd.Series(list(list(best_model_run[fields])[0]), index=pd.date_range(sim_start, sim_end))
    # Only necessary because spot_setup.evaluation() has a datetime. Thus both need a datetime.

    fig1 = plt.figure(1, figsize=(9, 5))
    plt.plot(results['like1'])
    plt.ylabel('NS-Eff')
    plt.xlabel('Iteration')
    if savefig:
        plt.savefig(dbname + '_sampling_plot.png')

    fig2 = plt.figure(figsize=(16, 9))
    ax = plt.subplot(1, 1, 1)
    ax.plot(best_simulation, color='black', linestyle='solid', label='Best objf.=' + str(bestobjf))
    ax.plot(spot_setup.evaluation(), 'r.', markersize=3, label='Observation data')
    plt.xlabel('Date')
    plt.ylabel('Discharge [mm d-1]')
    plt.legend(loc='upper right')
    if savefig:
        plt.savefig(dbname + '_best_run_plot.png')

    fig3 = plt.figure(figsize=(16, 9))
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
            label='data')  # Need to remove Timestamp from Evaluation to make comparable
    ax.set_ylim(0, 100)
    ax.set_xlim(0, len(spot_setup.evaluation()))
    ax.legend()
    if savefig:
        plt.savefig(dbname + '_par_uncertain_plot.png')

    return {'best_param': best_param, 'best_index': bestindex, 'best_model_run': best_model_run, 'best_objf': bestobjf,
            'best_simulation': best_simulation, 'param': spot_setup.param,
            'opt_iter': spot_setup.par_iter, 'sampling_plot': fig1, 'best_run_plot': fig2, 'par_uncertain_plot': fig3}
