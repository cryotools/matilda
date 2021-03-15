"""Glacier routine for the HBV model, lookup table generation from Seibert 2018"""

## Packages
from pathlib import Path; home = str(Path.home())
import pandas as pd
import numpy as np
## Files + Input
working_directory = "/Seafile/Ana-Lena_Phillip/data/HBV-Light/HBV-light_data/Glacier_No.1/"
input = home + working_directory + "Glacier_Routine_Data/GlacierProfile.txt"
output = home + working_directory + "Glacier_Routine_Data/"

glacier_profile = pd.read_csv(input, sep="\t", header=1) # Glacier Profile
area = pd.read_csv(input, sep="\t", nrows=1, header=None) # Area of catchment and glacier proportion
elevation_zones = glacier_profile["Elevation"].values.tolist()

# Preparing the variables
initial_area = glacier_profile["Area"] # per elevation band
initial_area_zones = glacier_profile.groupby("EleZone")["Area"].sum()
hi_initial = glacier_profile["WE"] # initial water equivalent of each elevation band
hi_k = glacier_profile["WE"] # hi_k is the updated water equivalent for each elevation zone, starts with initial values
ai = glacier_profile["Area"] # ai is the glacier area of each elevation zone, starts with initial values

lookup_table = pd.DataFrame()
lookup_table = lookup_table.append(initial_area, ignore_index=True)


## Pre-simulation
# 1. calculate total glacier mass in mm water equivalent: M = sum(ai * hi)
m = sum(glacier_profile["Area"]*glacier_profile["WE"])

# melt the glacier in steps of 1 percent
deltaM = -m/100

# 2. Normalize glacier elevations: Einorm = (Emax-Ei)/(Emax-Emin)
glacier_profile["norm_elevation"] = (glacier_profile["Elevation"].max() - glacier_profile["Elevation"]) / \
                                    (glacier_profile["Elevation"].max() - glacier_profile["Elevation"].min())

# 3. Apply deltaH parameterization: deltahi = (Einorm+a)^y + b*(Einorm+a) + c
# deltahi is the normalized (dimensionless) ice thickness change of elevation band i
# choose one of the three parameterizations from Huss et al. (2010) depending on glacier size
if area.iloc[0,0] * area.iloc[0,1] < 5:
    a = -0.3
    b = 0.6
    c = 0.09
    y = 2
elif area.iloc[0,0] * area.iloc[0,1] < 20:
    a = -0.05
    b = 0.19
    c = 0.01
    y = 4
else:
    a = -0.02
    b = 0.12
    c = 0
    y = 6

glacier_profile["delta_h"] = (glacier_profile["norm_elevation"] + a)**y + (b*(glacier_profile["norm_elevation"] + a))+c

## LOOP

ai_scaled = ai.copy()  # setting ai_scaled with the initial values


fs = deltaM / (sum(ai * glacier_profile["delta_h"]))  # a) initial ai

for _ in range(99):

    # 5. compute glacier geometry for reduced mass

    hi_k = hi_k + fs * glacier_profile["delta_h"]

    leftover = sum(pd.Series(np.where(hi_k < 0, hi_k, 0)) * ai)  # Calculate leftover (i.e. the 'negative' glacier volume)

    hi_k = pd.Series(np.where(hi_k < 0, np.nan, hi_k))  # Set those zones that have a negative weq to NaN to make sure they will be excluded from now on

    # 6. width scaling

    ai_scaled = ai * np.sqrt((hi_k / hi_initial))

    # 7. create lookup table

    # glacier area for each elevation band for 101 different mass situations (100 percent to 0 in 1 percent steps)

    lookup_table = lookup_table.append(ai_scaled, ignore_index=True)

    if sum(pd.Series(np.where(np.isnan(ai_scaled), 0, ai)) * glacier_profile["delta_h"]) == 0:
        ai_scaled = np.where(ai_scaled == 1, 1, 0)
    else:
        # Update fs (taking into account the leftover)
        fs = (deltaM + leftover) / sum(pd.Series(np.where(np.isnan(ai_scaled), 0, ai)) * glacier_profile["delta_h"])

lookup_table = lookup_table.fillna(0)
## Analysis
lookup_table.columns = glacier_profile["EleZone"]
lookup_table = lookup_table.groupby(level=0, axis=1).sum()

elezones_inital = lookup_table.iloc[0]

lookup_table = lookup_table / elezones_inital
lookup_table = round(lookup_table, 4)
lookup_table.iloc[-1] = 0

hbv_light = pd.read_csv(home + working_directory + "Python/Glacier_Run/Results/GlacierAreaLookupTable.txt", sep="\t")

#lookup_table_elezones.to_csv(output + "lookup_python.txt", index=None, header=True, sep="\t")

##

