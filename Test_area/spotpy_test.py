##
import spotpy                                                # Load the SPOT package into your working storage
from spotpy import analyser                                  # Load the Plotting extension
from spotpy.examples.spot_setup_rosenbrock import spot_setup # Import the two dimensional Rosenbrock example

# Give Monte Carlo algorithm the example setup and saves results in a RosenMC.csv file
sampler = spotpy.algorithms.mc(spot_setup(), dbname='RosenMC', dbformat='csv')
sampler.sample(100000)                # Sample 100.000 parameter combinations
results=sampler.getdata()             # Get the results of the sampler
spotpy.analyser.plot_parameterInteraction(results)
posterior=spotpy.analyser.get_posterior(results, percentage=10)
spotpy.analyser.plot_parameterInteraction(posterior)
print(spotpy.analyser.get_best_parameterset(results))

## Loading in the data


##
import numpy as np
from spotpy.parameter import Uniform
from spotpy.objectivefunctions import NSE
from spotpy.examples.hymod_python.hymod import hymod
import os

class spot_setup(object):
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
    CFMAX_ice = Uniform(low=1, high=10)
    CFMAX_snow = Uniform(low=1, high=10)
    SFCF = Uniform(low=0.4, high=1)
    CFR_ice = Uniform(low=0, high=0.1)
    CFR_snow = Uniform(low=0, high=0.1)
    CWH = Uniform(low=0, high=0.2)

    def __init__(self, df, obj_func=None):
        self.obj_func = obj_func
        df = df.resample("D").agg({"T2": 'mean', "RRR": 'sum'})
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

    def simulation(self):
        sim = hbv_simulation(self.Temp, self.Prec, self.Evap)
        return sim[366:]

    def evaluation(self):
        return self.Qobs[366:]

    def objectivefunction(self, simulation, evaluation, params=None):
        #SPOTPY expects to get one or multiple values back,
        #that define the performance of the model run
        if not self.obj_func:
            # This is used if not overwritten by user
            like = NSE(evaluation,simulation)
        else:
            #Way to ensure flexible spot setup class
            like = self.obj_func(evaluation,simulation)
        return like

