""" Debris cover, routine from Carenzo et al. (2016) """
##
from pathlib import Path; home = str(Path.home())
import numpy as np
import pandas as pd

## Debris Enhanced Temperature-Index (DETI) model
# if T > TT
# m = TF * T (i − lagT) + SRF  * (1 − α) * I (i − lagI)
# if T <= TT 0
# m = hourly melt rate in mm w.e.
# T = Temperature, α = albedo,  I = incoming shortwave radiation, i = timestep (?)
# empirical factors SRF and TF , lag parameters lagT and lag I --> fixed table depending on debris thickness

# for different elevation zones?
# table with elevation zone, glacier area, debris thickness per zone, albedo?

df = pd.read_csv(home + "/Seafile/Ana-Lena_Phillip/data/matilda/Test_area/Debris_test/test_DETI.csv", parse_dates=['TIMESTAMP'], index_col='TIMESTAMP')
glacier_table = pd.read_csv(home + "/Seafile/Ana-Lena_Phillip/data/matilda/Test_area/Debris_test/test_DETI2.csv")
glacier_area = glacier_table["area"].sum()
time = df.index
temp = df["T2"] - 273.15
shortwave = df["shortwave"]

def deti_melt(time, temp, shortwave, glacier_table, glacier_area):
    debris_thickness = glacier_table["debris_thickness"].values
    albedo = glacier_table["albedo"].values
    glacier_area_zones = glacier_table["area"].values

    total_melt = [0] * len(time)

    for debris, alb, area in zip(debris_thickness, albedo, glacier_area_zones):
        # get parameters
        if debris <= 0.05:
            lagT, lagI, TF, SRF = 0, 0, 0.0984, 0.0044
        if 0.05 < debris <= 0.1:
            lagT, lagI, TF, SRF = 0, 1, 0.0660, 0.0023
        if 0.1 < debris <= 0.2:
            lagT, lagI, TF, SRF = 3, 3, 0.0456, 0.0009
        if 0.2 < debris <= 0.23:
            lagT, lagI, TF, SRF = 3, 4, 0.0438, 0.0006
        if 0.23 < debris <= 0.3:
            lagT, lagI, TF, SRF = 5, 5, 0.0392, 0.0002
        if 0.3 < debris <= 0.4:
            lagT, lagI, TF, SRF = 7, 7, 0.0334, 0.0001
        if debris <= 0.4:
            lagT, lagI, TF, SRF = 10, 11, 0.0265, 0

        # is i really the hour of the day?
        m = TF * temp * (time.hour - lagT) + SRF * (1 - alb) * shortwave * (time.hour - lagI)
        m = m * area/glacier_area
        total_melt = total_melt + m
    return total_melt

melt = deti_melt(time, temp, shortwave, glacier_table, glacier_area)

## our melt model based on the pypdd
def melt_rates(snow, pdd):
    # compute a potential snow melt
    pot_snow_melt = CFMAX_snow * pdd
    # effective snow melt can't exceed amount of snow
    snow_melt = np.minimum(snow, pot_snow_melt)
    # ice melt is proportional to excess snow melt
    ice_melt = (pot_snow_melt - snow_melt) * CFMAX_ice / CFMAX_snow
    # return melt rates
    return (snow_melt, ice_melt)



