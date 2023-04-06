import pandas as pd
from pathlib import Path
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import spotpy  # Load the SPOT package into your working storage
from datetime import date, datetime
import HydroErr as he
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import mae, rmse
from spotpy import analyser  # Load the Plotting extension
home = str(Path.home())
# sys.path.append(home + '/Ana-Lena_Phillip/data/tests_and_tools/Test_area/SPOTPY')
# import mspot
from matilda.core import matilda_simulation, matilda_parameter, matilda_preproc, create_lookup_table, updated_glacier_melt


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


def spot_setup(set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D", lat=None, area_cat=None,
               area_glac=None, ele_dat=None, ele_glac=None, ele_cat=None, soi = None, glacier_profile=None,
               elev_rescaling=True, target_mb = None,
            lr_temp_lo=-0.01, lr_temp_up=-0.003,
            lr_prec_lo=0, lr_prec_up=0.002,
            BETA_lo=1, BETA_up=6,
            CET_lo=0, CET_up=0.3,
            FC_lo=50, FC_up=500,
            K0_lo=0.01, K0_up=0.4,
            K1_lo=0.01, K1_up=0.4,
            K2_lo=0.001, K2_up=0.15,
            LP_lo=0.3, LP_up=1,
            MAXBAS_lo=2,  MAXBAS_up=7,
            PERC_lo=0, PERC_up=3,
            UZL_lo=0, UZL_up=500,
            PCORR_lo=0.5, PCORR_up=2,
            TT_snow_lo=-1.5, TT_snow_up=1.5,
            TT_diff_lo=0.5, TT_diff_up=2.5,
            CFMAX_ice_lo=1.2, CFMAX_ice_up=12,
            CFMAX_rel_lo=1.2, CFMAX_rel_up=2.5,
            SFCF_lo=0.4, SFCF_up=1,
            CWH_lo=0, CWH_up=0.2,
            AG_lo=0, AG_up=1,
            RFS_lo=0.05, RFS_up=0.25,

            interf=4, freqst=2):

    class spot_setup:
        # defining all parameters and the distribution
        param = lr_temp, lr_prec, BETA, CET, FC, K0, K1, K2, LP, MAXBAS, PERC, UZL, PCORR, \
                TT_snow, TT_diff, CFMAX_ice, CFMAX_rel, SFCF, CWH, AG, RFS = [
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
            Uniform(low=TT_diff_lo, high=TT_diff_up),  # TT_diff
            Uniform(low=CFMAX_ice_lo, high=CFMAX_ice_up), # CFMAX_ice
            Uniform(low=CFMAX_rel_lo, high=CFMAX_rel_up),  # CFMAX_rel
            Uniform(low=SFCF_lo, high=SFCF_up),  # SFCF
            Uniform(low=CWH_lo, high=CWH_up),  # CWH
            Uniform(low=AG_lo, high=AG_up),  # AG
            Uniform(low=RFS_lo, high=RFS_up),  # RFS
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
                sim = matilda_simulation(self.Input, obs=self.obs,
                                                 output=None, set_up_start=set_up_start, set_up_end=set_up_end,
                                                 sim_start=sim_start, sim_end=sim_end, freq=freq, lat=lat, soi=soi,
                                                 area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat,
                                                 ele_glac=ele_glac, ele_cat=ele_cat, plots=False, warn=False,
                                                 glacier_profile=glacier_profile, elev_rescaling=elev_rescaling,

                                                 lr_temp=x.lr_temp, lr_prec=x.lr_prec,
                                                 BETA=x.BETA, CET=x.CET, FC=x.FC, K0=x.K0, K1=x.K1, K2=x.K2, LP=x.LP,
                                                 MAXBAS=x.MAXBAS, PERC=x.PERC, UZL=x.UZL, PCORR=x.PCORR,
                                                 TT_snow=x.TT_snow, TT_diff=x.TT_diff, CFMAX_ice=x.CFMAX_ice,
                                                 CFMAX_rel=x.CFMAX_rel, SFCF=x.SFCF, CWH=x.CWH, AG=x.AG, RFS=x.RFS)
            if target_mb is None:
                return sim[0].total_runoff
            else:
                return [sim[0].total_runoff, sim[5].smb_water_year.mean()]

        def evaluation(self):
            obs_preproc = self.obs.copy()
            obs_preproc.set_index('Date', inplace=True)
            obs_preproc.index = pd.to_datetime(obs_preproc.index)
            obs_preproc = obs_preproc[sim_start:sim_end]
            # Changing the input unit from m³/s to mm.
            obs_preproc["Qobs"] = obs_preproc["Qobs"] * 86400 / (area_cat * 1000000) * 1000
            # To daily resolution
            obs_preproc = obs_preproc.resample("D").agg(pd.Series.sum, skipna=False)
            # Omit everything outside the specified season of interest (soi)
            if soi is not None:
                obs_preproc = obs_preproc[obs_preproc.index.month.isin(range(soi[0], soi[1] + 1))]
            # Expanding the observation period full years filling up with NAs
            idx_first = obs_preproc.index.year[1]
            idx_last = obs_preproc.index.year[-1]
            idx = pd.date_range(start=date(idx_first, 1, 1), end=date(idx_last, 12, 31), freq='D',
                                name=obs_preproc.index.name)
            obs_preproc = obs_preproc.reindex(idx)
            obs_preproc = obs_preproc.fillna(np.NaN)

            if target_mb is None:
                return obs_preproc.Qobs
            else:
                return [obs_preproc.Qobs, target_mb]

        def objectivefunction(self, simulation, evaluation, params=None):
            # SPOTPY expects to get one or multiple values back,
            # that define the performance of the model run
            if target_mb is not None:
                obj2 = abs(evaluation[1] - simulation[1])
                simulation = simulation[0]
                evaluation = evaluation[0]

            # Crop both timeseries to same periods without NAs
            sim_new = pd.DataFrame()
            sim_new['mod'] = pd.DataFrame(simulation)
            sim_new['obs'] = evaluation
            clean = sim_new.dropna()

            simulation_clean = clean['mod']
            evaluation_clean = clean['obs']

            if not self.obj_func:
                # This is used if not overwritten by user
                # obj1 = kge(evaluation_clean, simulation_clean)          # SPOTPY internal kge
                obj1 = he.kge_2012(simulation_clean, evaluation_clean, remove_zero=True) # same as MATILDA
            else:
                # Way to ensure flexible spot setup class
                obj1 = self.obj_func(evaluation_clean, simulation_clean)

            if target_mb is None:
                return obj1
            else:
                return [obj1, obj2]

    return spot_setup




def winter(year, data):
    data = data[data.index == year]
    winter = slice(data.BEGIN_PERIOD.squeeze(), data.END_WINTER.squeeze())
    return winter

def summer(year, data):
    data = data[data.index == year]
    summer = slice(data.END_WINTER.squeeze(), data.END_PERIOD.squeeze())
    return summer

def annual(year, data):
    data = data[data.index == year]
    summer = slice(data.BEGIN_PERIOD.squeeze(), data.END_PERIOD.squeeze())
    return summer

def spot_setup_glacier(set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D", lat=None, area_cat=None,
               area_glac=None, ele_dat=None, ele_glac=None, ele_cat=None, soi=None, glacier_profile=None, obs_type="annual",
            lr_temp_lo=-0.01, lr_temp_up=-0.003,
            lr_prec_lo=0, lr_prec_up=0.002,
            PCORR_lo=0.5, PCORR_up=2,
            TT_snow_lo=-1.5, TT_snow_up=1.5,
            TT_diff_lo=0.5, TT_diff_up=2.5,
            CFMAX_ice_lo=1.2, CFMAX_ice_up=12,
            CFMAX_rel_lo=1.2, CFMAX_rel_up=2.5,
            SFCF_lo=0.4, SFCF_up=1,
            RFS_lo=0.05, RFS_up=0.25,

            interf=4, freqst=2):

    class spot_setup:
        # defining all parameters and the distribution
        param = lr_temp, lr_prec, PCORR, \
                TT_snow, TT_diff, CFMAX_ice, CFMAX_rel, SFCF, RFS = [
            Uniform(low=lr_temp_lo, high=lr_temp_up),  # lr_temp
            Uniform(low=lr_prec_lo, high=lr_prec_up),  # lr_prec
            Uniform(low=PCORR_lo, high=PCORR_up),  # PCORR
            Uniform(low=TT_snow_lo, high=TT_snow_up),  # TT_snow
            Uniform(low=TT_diff_lo, high=TT_diff_up),  # TT_diff
            Uniform(low=CFMAX_ice_lo, high=CFMAX_ice_up), # CFMAX_snow
            Uniform(low=CFMAX_rel_lo, high=CFMAX_rel_up),  # CFMAX_rel
            Uniform(low=SFCF_lo, high=SFCF_up),  # SFCF
            Uniform(low=RFS_lo, high=RFS_up),  # RFS
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
                parameter = matilda_parameter(self.Input,
                                         set_up_start=set_up_start, set_up_end=set_up_end,
                                         sim_start=sim_start, sim_end=sim_end, freq=freq, lat=lat, soi=soi,
                                         area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat,
                                         ele_glac=ele_glac, ele_cat=None,

                                         lr_temp=x.lr_temp, lr_prec=x.lr_prec, PCORR=x.PCORR,
                                         TT_snow=x.TT_snow, TT_diff=x.TT_diff, CFMAX_ice=x.CFMAX_ice,
                                         CFMAX_rel=x.CFMAX_rel, SFCF=x.SFCF, RFS=x.RFS)

                df_preproc = matilda_preproc(self.Input, parameter)
                lookup_table = create_lookup_table(glacier_profile, parameter)
                output_DDM = updated_glacier_melt(df_preproc, lookup_table, glacier_profile, parameter)[0]
                sim = output_DDM.DDM_smb

            return sim

        def evaluation(self):
            obs_preproc = self.obs.copy()
            obs_preproc.set_index('YEAR', inplace=True)
            obs_preproc.index = pd.to_datetime(obs_preproc.index)

            return obs_preproc

        def objectivefunction(self, simulation, evaluation, params=None):
            # Aggregate MBs to fit the calibration data
            sim = []
            obs = []
            sim_new = pd.DataFrame()
            if obs_type == "winter":
                for i in evaluation.index:
                    mb = simulation[winter(i, evaluation)].sum()
                    sim.append(mb)
                    mb_obs = evaluation[evaluation.index == i].WINTER_BALANCE.squeeze()
                    obs.append(mb_obs)

            if obs_type == "summer":
                for i in evaluation.index:
                    mb = simulation[summer(i, evaluation)].sum()
                    sim.append(mb)
                    mb_obs = evaluation[evaluation.index == i].SUMMER_BALANCE.squeeze()
                    obs.append(mb_obs)

            if obs_type == "annual":
                for i in evaluation.index:
                    mb_sim = simulation[annual(i, evaluation)].sum()
                    sim.append(mb_sim)
                    mb_obs = evaluation[evaluation.index == i].ANNUAL_BALANCE.squeeze()
                    obs.append(mb_obs)

            # Crop both timeseries to same periods without NAs
            sim_new['mod'] = pd.DataFrame(sim)
            sim_new['obs'] = pd.DataFrame(obs)

            clean = sim_new.dropna()

            simulation_clean = clean['mod']
            evaluation_clean = clean['obs']

            if not self.obj_func:
                # This is used if not overwritten by user
                like = mae(evaluation_clean, simulation_clean)
                print('MAE: ' + str(like))
            else:
                # Way to ensure flexible spot setup class
                like = self.obj_func(evaluation_clean, simulation_clean)
            return like

    return spot_setup



def analyze_results(sampling_data, obs, algorithm, obj_dir="maximize", fig_path = None, dbname='mspot_results',
                    glacier_only=False, target_mb=None):

    if isinstance(obs, str):
        if glacier_only:
            obs = pd.read_csv(obs)
        else:
            obs = pd.read_csv(obs, index_col='Date', parse_dates=['Date'])

    if isinstance(sampling_data, str):
        if sampling_data.endswith('.csv'):
            sampling_data = sampling_data[:len(sampling_data)-4]
        results = spotpy.analyser.load_csv_results(sampling_data)
    else:
        results = sampling_data.getdata()

    maximize = ["abc", "dds", "demcz", "dream", "rope", "sa", "fscabc", "mcmc", "mle"]
    minimize = ["nsgaii", "padds", "sceua"]
    both = ["fast", "lhs", "mc"]

    if algorithm in maximize:
        best_param = spotpy.analyser.get_best_parameterset(results)
    elif algorithm in minimize:
        best_param = spotpy.analyser.get_best_parameterset(results, maximize=False)
    elif algorithm in both:
        print("WARNING: The selected algorithm " + algorithm + " can either maximize or minimize the objective function."
              " You can specify the direction by passing obj_dir to analyze_results(). The default is 'maximize'.")
        if obj_dir == "maximize":
            best_param = spotpy.analyser.get_best_parameterset(results)
        elif obj_dir == "minimize":
            best_param = spotpy.analyser.get_best_parameterset(results, maximize=False)
        else:
            print("Invalid argument for obj_dir. Choose 'minimize' or 'maximize'.")
            return
    else:
        print("Invalid argument for algorithm. Available algorithms: ['abc', 'dds', 'demcz', 'dream', 'rope', 'sa',"
              "'fscabc', 'mcmc', 'mle', 'nsgaii', 'padds', 'sceua', 'fast', 'lhs', 'mc']")
        return

    par_names = spotpy.analyser.get_parameternames(best_param)
    param_zip = zip(par_names, best_param[0])
    best_param = dict(param_zip)

    if glacier_only:
        bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)  # Run with lowest MAE
        best_model_run = results[bestindex]

    elif target_mb is not None:
        if obj_dir == "minimize":
            bestindex, bestobjf = spotpy.analyser.get_minlikeindex(results)
        else:
            bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(results)
        best_model_run = results[bestindex]

    else:
        sim_start = obs.index[0]
        sim_end = obs.index[-1]
        bestindex, bestobjf = spotpy.analyser.get_maxlikeindex(results)  # Run with highest KGE
        best_model_run = results[bestindex]
        fields = [word for word in best_model_run.dtype.names if word.startswith('sim')]
        best_simulation = pd.Series(list(list(best_model_run[fields])[0]), index=pd.date_range(sim_start, sim_end))
        # Only necessary because obs has a datetime. Thus, both need a datetime.
        
    if not glacier_only and target_mb is None:
        fig1 = plt.figure(1, figsize=(9, 5))
        plt.plot(results['like1'])
        plt.ylabel('KGE')
        plt.xlabel('Iteration')
        if fig_path is not None:
            plt.savefig(fig_path + '/' + dbname + '_sampling_plot.png')

        fig2 = plt.figure(figsize=(16, 9))
        ax = plt.subplot(1, 1, 1)
        ax.plot(best_simulation, color='black', linestyle='solid', label='Best objf.=' + str(bestobjf))
        ax.plot(obs, 'r.', markersize=3, label='Observation data')
        plt.xlabel('Date')
        plt.ylabel('Discharge [mm d-1]')
        plt.legend(loc='upper right')
        if fig_path is not None:
            plt.savefig(fig_path + '/' + dbname + '_best_run_plot.png')

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
        ax.plot(np.array(obs), 'r.',
                label='data')  # Need to remove Timestamp from Evaluation to make comparable
        # ax.set_ylim(0, 100)
        ax.set_xlim(0, len(obs))
        ax.legend()
        if fig_path is not None:
            plt.savefig(fig_path + '/' + dbname + '_par_uncertain_plot.png')

        return {'best_param': best_param, 'best_index': bestindex, 'best_model_run': best_model_run, 'best_objf': bestobjf,
                'best_simulation': best_simulation,
                'sampling_plot': fig1, 'best_run_plot': fig2, 'par_uncertain_plot': fig3}

    else:
        return {'best_param': best_param, 'best_index': bestindex, 'best_model_run': best_model_run,
                'best_objf': bestobjf}

def psample(df, obs, rep=10, output = None, dbname='matilda_par_smpl', dbformat=None, obj_func=None, opt_iter=False, fig_path=None, #savefig=False,
            set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D", lat=None, area_cat=None,
            area_glac=None, ele_dat=None, ele_glac=None, ele_cat=None, soi=None, glacier_profile=None,
            interf=4, freqst=2, parallel=False, cores=2, save_sim=True, elev_rescaling=True,
            glacier_only=False, obs_type="annual", target_mb=None,
            algorithm='sceua', obj_dir="maximize", **kwargs):

    cwd = os.getcwd()
    if output is not None:
        os.chdir(output)

    # Setup model class:

    if glacier_only:
        setup = spot_setup_glacier(set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start, sim_end=sim_end,
                           freq=freq, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, ele_glac=ele_glac,
                           ele_cat=ele_cat, lat=lat, soi=soi, interf=interf, freqst=freqst,
                           glacier_profile=glacier_profile, obs_type=obs_type,
                           **kwargs)

    else:
        setup = spot_setup(set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start, sim_end=sim_end,
                        freq=freq, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, ele_glac=ele_glac,
                        ele_cat=ele_cat, lat=lat, soi=soi, interf=interf, freqst=freqst, glacier_profile=glacier_profile,
                        elev_rescaling=elev_rescaling, target_mb=target_mb,
                        **kwargs)

    psample_setup = setup(df, obs, obj_func)  # Define custom objective function using obj_func=
    alg_selector = {'mc': spotpy.algorithms.mc, 'sceua': spotpy.algorithms.sceua, 'mcmc': spotpy.algorithms.mcmc,
                    'mle': spotpy.algorithms.mle, 'abc': spotpy.algorithms.abc, 'sa': spotpy.algorithms.sa,
                    'dds': spotpy.algorithms.dds, 'demcz': spotpy.algorithms.demcz,
                    'dream': spotpy.algorithms.dream, 'fscabc': spotpy.algorithms.fscabc,
                    'lhs': spotpy.algorithms.lhs, 'padds': spotpy.algorithms.padds,
                    'rope': spotpy.algorithms.rope, 'fast': spotpy.algorithms.fast,
                    'nsgaii': spotpy.algorithms.nsgaii}

    if target_mb is not None:           # Format errors in database csv when saving simulations
        save_sim = False

    if parallel:
        sampler = alg_selector[algorithm](psample_setup, dbname=dbname, dbformat=dbformat, parallel='mpi',
                                              optimization_direction=obj_dir, save_sim=save_sim)
        if algorithm == 'mc' or algorithm == 'lhs' or algorithm == 'fast' or algorithm == 'rope':
            sampler.sample(rep)
        elif algorithm == 'sceua':
            sampler.sample(rep, ngs=cores)
        elif algorithm == 'demcz':
            sampler.sample(rep, nChains=cores)
        else:
            print('ERROR: The selected algorithm is ineligible for parallel computing.'
                  'Either select a different algorithm (mc, lhs, fast, rope, sceua or demcz) or set "parallel = False".')
            return
    else:
        sampler = alg_selector[algorithm](psample_setup, dbname=dbname, dbformat=dbformat, save_sim=save_sim,
                                          optimization_direction=obj_dir)
        if opt_iter:
            if yesno("\n******** WARNING! Your optimum # of iterations is {0}. "
                     "This may take a long time.\n******** Do you wish to proceed".format(psample_setup.par_iter)):
                sampler.sample(psample_setup.par_iter)  # ideal number of reps = psample_setup.par_iter
            else:
                return
        else:
            sampler.sample(rep)

    # Change dbformat to None for short tests but to 'csv' or 'sql' to avoid data loss in case off long calculations.

    if target_mb is None:
        psample_setup.evaluation().to_csv(dbname + '_observations.csv')
    else:
        psample_setup.evaluation()[0].to_csv(dbname + '_observations.csv')


    if not parallel:

        if target_mb is None:
            results = analyze_results(sampler, psample_setup.evaluation(), algorithm=algorithm, obj_dir=obj_dir,
                                  fig_path=fig_path, dbname=dbname, glacier_only=glacier_only)
        else:
            results = analyze_results(sampler, psample_setup.evaluation()[0], algorithm=algorithm, obj_dir=obj_dir,
                                  fig_path=fig_path, dbname=dbname, glacier_only=glacier_only, target_mb=target_mb)

        return results

    os.chdir(cwd)


def load_parameters(path, algorithm, obj_dir="maximize", glacier_only=False):
    sampling_csv = path
    sampling_obs = path.split('.')[0] + '_observations.csv'
    results = analyze_results(sampling_csv, sampling_obs, algorithm, obj_dir, glacier_only=glacier_only)

    return results['best_param']


def get_par_bounds(path, threshold=10, percentage=True, drop=[]):
    result_path = path
    results = spotpy.analyser.load_csv_results(result_path)
    # Get parameters of best model runs
    if percentage:
        best = spotpy.analyser.get_posterior(results, maximize=False, percentage=threshold)      # get best xx% runs
    else:
        best = results[np.where(results["like1"] <= threshold)]            # get all runs below a treshhold
    params = spotpy.analyser.get_parameters(best)
    # Write min and max parameter values of best model runs to dictionary
    par_bounds = {}
    for i in spotpy.analyser.get_parameter_fields(best):
        p = params[i]
        par_bounds[i.split('par')[1] + '_lo'] = min(p)
        par_bounds[i.split('par')[1] + '_up'] = max(p)
    for i in drop:
        del par_bounds[i + '_lo']
        del par_bounds[i + '_up']

    return par_bounds


def dict2bounds(p_dict, drop=[]):
    for i in drop:
        del p_dict[i]
    p = {**dict(zip([i + '_lo' for i in p_dict.keys()], p_dict.values())),
         **dict(zip([i + '_up' for i in p_dict.keys()], p_dict.values()))}
    return p


def scaled_pdd(data, elev, lr):
    s = data + elev * lr
    pdd = np.where(s > 0, s, 0)
    return pdd


def scaled_ndd(data, elev, lr):
    s = data + elev * lr
    ndd = np.where(s < 0, 1, 0)
    return ndd