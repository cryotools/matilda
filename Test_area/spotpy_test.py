##
import pandas as pd
from datetime import date
import spotpy  # Load the SPOT package into your working storage
from spotpy import analyser  # Load the Plotting extension
from spotpy.examples.spot_setup_rosenbrock import spot_setup  # Import the two dimensional Rosenbrock example
from pathlib import Path

home = str(Path.home())

# ## SPOTPY example from the read the docs website
# # Give Monte Carlo algorithm the example setup and saves results in a RosenMC.csv file
# sampler_ros = spotpy.algorithms.mc(spot_setup(), dbname='RosenMC', dbformat='csv')
# sampler_ros.sample(100000)  # Sample 100.000 parameter combinations
# results = sampler_ros.getdata()  # Get the results of the sampler
# spotpy.analyser.plot_parameterInteraction(results)
# posterior = spotpy.analyser.get_posterior(results, percentage=10)
# spotpy.analyser.plot_parameterInteraction(posterior)
# print(spotpy.analyser.get_best_parameterset(results))

## Loading in the data
# Making sure that the data has the same length and frequency since I excluded the steps in the SPOTPY run
df = pd.read_csv(
    home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv")
obs = pd.read_csv(
    home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/runoff_bashkaindy_04_2019-11_2020_temp_limit.csv")
start = "2019-01-01"
end = "2020-11-01"

df.set_index('TIMESTAMP', inplace=True)
df.index = pd.to_datetime(df.index)
df = df[start:end]
df = df.resample("D").agg({"T2": 'mean', "RRR": 'sum'})

obs.set_index("Date", inplace=True)
obs.index = pd.to_datetime(obs.index)
obs = obs.resample("D").sum()
# expanding the observation period to the whole one year, filling the NAs with 0
idx_first = obs.index.year[1]
idx_last = obs.index.year[-1]
idx = pd.date_range(start=date(idx_first, 1, 1), end=date(idx_last, 12, 31), freq='D', name=obs.index.name)
obs = obs.reindex(idx)
obs = obs.fillna(0)
obs = obs[start:end]

df["Qobs"] = obs["Qobs"]  # df will be the input dataframe for SPOTPY

## Setting up SPOTPY
import numpy as np
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import nashsutcliffe

import spotpy_hbv

# creating a spotpy setup class
class spot_setup(object):
    # defining all parameters and the distribution
    BETA = Uniform(low=1, high=6)
    CET = Uniform(low=0, high=0.3)
    FC = Uniform(low=50, high=500)
    K0 = Uniform(low=0.01, high=0.4)
    K1 = Uniform(low=0.01, high=0.4)
    K2 = Uniform(low=0.001, high=0.15)
    LP = Uniform(low=0.3, high=1)
    MAXBAS = Uniform(low=2, high=7)
    PERC = Uniform(low=0, high=3)
    UZL = Uniform(low=0, high=500)
    PCORR = Uniform(low=0.5, high=2)
    TT_snow = Uniform(low=-1.5, high=2.5)
    TT_rain = Uniform(low=-1.5, high=2.5)
    CFMAX_snow = Uniform(low=1, high=10)
    SFCF = Uniform(low=0.4, high=1)
    CFR_snow = Uniform(low=0, high=0.1)
    CWH = Uniform(low=0, high=0.2)

    # loading in all the data. Not 100% sure if this is correct
    def __init__(self, df, obj_func=None):
        self.obj_func = obj_func
        extra_rad = 27.086217947590317
        latent_heat_flux = 2.45
        water_density = 1000

        self.Temp = df["T2"]
        if self.Temp[1] >= 100:
            self.Temp = self.Temp - 273.15
        self.Prec = df["RRR"]
        df["PE"] = np.where((self.Temp) + 5 > 0, ((extra_rad / (water_density * latent_heat_flux)) * \
                                                  ((self.Temp) + 5) / 100) * 1000, 0)
        self.Evap = df["PE"]
        self.Qobs = df["Qobs"]

    # starting the simulation. The function hbv_simulation is in the spotpy_hbv script
    def simulation(self):
        sim = hbv_simulation(self.Temp, self.Prec, self.Evap, self.BETA, self.CET, self.FC, self.K0, self.K1, self.K2,
                             self.LP, self.MAXBAS, self.PERC, self.UZL, self.PCORR, self.TT_snow, self.TT_rain,
                             self.CFMAX_snow, self.SFCF, self.CFR_snow, self.CWH)

        return sim[366:]  # excludes the first year as a spinup period

    def evaluation(self):
        return self.Qobs[366:]

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = nashsutcliffe(evaluation, simulation)
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like


##
rep = 100
spot_setup = spot_setup(df)  # setting up the model. Normally the brackets should be empty but we need the df argument here
sampler = spotpy.algorithms.mc(spot_setup, dbname='mc_hbv', dbformat='csv')  # links the setup to the Monte Carlo algorithm
sampler.sample(rep)  # runs the algorith. This step doesn't work
