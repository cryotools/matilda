"""Glacier routine for the HBV model, lookup table generation from Seibert 2018"""

## Packages
import pandas as pd
import numpy as np
## Files + Input
input = ".../GlacierProfile.txt"

glacier_profile = pd.read_csv(input, sep="\t", header=1) # Glacier Profile
area = pd.read_csv(input, sep="\t", nrows=1, header=None) # Area of catchment and glacier proportion
elevation_zones = glacier_profile["Elevation"].values.tolist()

# Preparing the variables
initial_area = glacier_profile["Area"]
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
ai_scaled = ai.copy() # setting ai_scaled with the initial values
deltaM_add = (glacier_profile["delta_h"]/sum(glacier_profile["delta_h"]))*deltaM
# we specify the percentage of every elevation band's deltaH factor and multiply it with deltaM to get the deltaM per
# elevation band

for _ in range(100):
    # add remaining melt to deltaM when the elevation zone reaches 0
    # the idea here is to take deltaM from every elevation band where ai_scaled is 0, then get the sum of it to add it
    # to the deltaM within the loop itself
    deltaM_zone = pd.Series((np.where(ai_scaled.isna(), deltaM_add, 0)))
    deltaM_zone[0] = 0
    deltaM_zone1 = sum(deltaM_zone)

    # 4. scaling factor to scale dimensionless deltah
    #fs = deltaM / (sum(ai_scaled*glacier_profile["delta_h"]))
    fs = (deltaM + deltaM_zone1) / sum(ai * glacier_profile["delta_h"]) # add the deltaM addition of the zone reached 0

    # 5. compute glacier geometry for reduced mass
    hi_k = hi_k + fs*glacier_profile["delta_h"]
    # 6. width scaling
    ai_scaled = ai * np.sqrt((hi_k/hi_initial))
    #ai_scaled = pd.Series(np.where(np.isnan(ai_scaled), 0, ai_scaled))
    # 7. create lookup table
    # glacier area for each elevation band for 101 different mass situations (100 percent to 0 in 1 percent steps)
    lookup_table = lookup_table.append(ai_scaled, ignore_index=True)

lookup_table = lookup_table.fillna(0)

## Analysis
lookup_table.columns = elevation_zones
lookup_table.sum(axis=1)

lookup_table_elezones = pd.DataFrame(columns=["3700", "3800", "3900", "4000", "4100", "4200", "4400"])
lookup_table_elezones["3700"] = lookup_table.iloc[:,0]
lookup_table_elezones["3800"] = lookup_table.iloc[:,1]
lookup_table_elezones["3900"] = lookup_table.iloc[:,2] + lookup_table.iloc[:,3]
lookup_table_elezones["4000"] = lookup_table.iloc[:,4] + lookup_table.iloc[:,5]
lookup_table_elezones["4100"] = lookup_table.iloc[:,6] + lookup_table.iloc[:,7]
lookup_table_elezones["4200"] = lookup_table.iloc[:,8] + lookup_table.iloc[:,9]
lookup_table_elezones["4400"] = lookup_table.iloc[:,10]

elezones_inital = lookup_table_elezones.iloc[0]

lookup_table_elezones = lookup_table_elezones / elezones_inital
lookup_table_elezones = round(lookup_table_elezones, 4)
lookup_table_elezones.iloc[-1] = 0 # setting the last step of the table to 0