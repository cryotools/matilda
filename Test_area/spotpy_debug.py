##
import pandas as pd
from datetime import date
import spotpy  # Load the SPOT package into your working storage
import numpy as np
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import nashsutcliffe
from spotpy import analyser  # Load the Plotting extension
from spotpy.examples.spot_setup_rosenbrock import spot_setup  # Import the two dimensional Rosenbrock example
from pathlib import Path
from spotpy_hbv import hbv_simulation

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

extra_rad = 27.086217947590317
latent_heat_flux = 2.45
water_density = 1000

df["T2"] = df["T2"] - 273.15
df["PE"] = np.where((df["T2"]) + 5 > 0, ((extra_rad / (water_density * latent_heat_flux)) * \
                                          ((df["T2"]) + 5) / 100) * 1000, 0)


## Setting up SPOTPY

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
    TT_snow = Uniform(low=-1.5, high=2.5)       # Gehen negative Werte?? Warum hier Celsius?
    TT_rain = Uniform(low=-1.5, high=2.5)       # Gehen negative Werte?? Warum hier Celsius?
    CFMAX_snow = Uniform(low=1, high=10)
    SFCF = Uniform(low=0.4, high=1)
    CFR_snow = Uniform(low=0, high=0.1)
    CWH = Uniform(low=0, high=0.2)

    # loading in all the data. Not 100% sure if this is correct
    def __init__(self, df, obj_func=None):
        self.obj_func = obj_func
        self.Temp = df["T2"]
        self.Prec = df["RRR"]
        self.Evap = df["PE"]
        self.Qobs = df["Qobs"]

    # starting the simulation. The function hbv_simulation is in the spotpy_hbv script
    def simulation(self, x):
        sim = hbv_simulation(self.Temp, self.Prec, self.Evap, x.BETA, x.CET, x.FC, x.K0, x.K1, x.K2,
                             x.LP, x.MAXBAS, x.PERC, x.UZL, x.PCORR, x.TT_snow, x.TT_rain,
                             x.CFMAX_snow, x.SFCF, x.CFR_snow, x.CWH)
        return sim[366:]  # excludes the first year as a spinup period

    def evaluation(self):
                                        
        return self.Qobs[366:]

    def objectivefunction(self, simulation, evaluation, params=None):
        # SPOTPY expects to get one or multiple values back,
        # that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = nashsutcliffe(evaluation, simulation)        # In den Beispielen ist hier ein Minus davor?!
        else:
            # Way to ensure flexible spot setup class
            like = self.obj_func(evaluation, simulation)
        return like

# test = spot_setup(df)
# spot_setup.simulation(test)
##
rep = 1000
spot_setup = spot_setup(df)  # setting up the model. Normally the brackets should be empty but we need the df argument here     Was soll "normalerweise" heissen?
sampler = spotpy.algorithms.mc(spot_setup, dbname='mc_hbv', dbformat='csv')  # links the setup to the Monte Carlo algorithm
sampler.sample(rep)  # runs the algorithm.

results = sampler.getdata()  # Get the results of the sampler
# spotpy.analyser.plot_parameterInteraction(results)
posterior = spotpy.analyser.get_posterior(results, percentage=10)
# spotpy.analyser.plot_parameterInteraction(posterior)
print(spotpy.analyser.get_best_parameterset(results))

res = pd.read_csv(
    home + "/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/mc_hbv.csv")

res = res.sort_values(['like1'])