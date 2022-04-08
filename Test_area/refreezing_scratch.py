##
import numpy as np
import xarray as xr
from pathlib import Path
import pandas as pd
from MATILDA_slim import MATILDA

## Prepare test df:
home = str(Path.home()) + '/Seafile'
wd = home + '/EBA-CA/Papers/No1_Kysylsuu_Bash-Kaingdy/data'
input_path = wd + "/input/kyzylsuu"
t2m_path = "/met/era5l/t2m_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
tp_path = "/met/era5l/tp_era5l_adjust_42.516-79.0167_1982-01-01-2020-12-31.csv"
t2m = pd.read_csv(input_path + t2m_path)
tp = pd.read_csv(input_path + tp_path)
df = pd.concat([t2m, tp.tp], axis=1)
df.rename(columns={'time': 'TIMESTAMP', 't2m': 'T2','tp':'RRR'}, inplace=True)

parameter = MATILDA.MATILDA_parameter(df, set_up_start='1982-01-01 00:00:00', set_up_end='1984-12-31 23:00:00',
                                      sim_start='1985-01-01 00:00:00', sim_end='1989-12-31 23:00:00', freq="D",
                                      area_cat=315.694, area_glac=32.51, lat=42.33, warn=True,
                                      ele_dat=2550, ele_glac=4074, ele_cat=3225)

df_preproc = MATILDA.MATILDA_preproc(df, parameter)

input_df_glacier, input_df_catchment = MATILDA.glacier_elevscaling(df_preproc, parameter)
ds = MATILDA.calculate_PDD(input_df_glacier)

# ds.to_netcdf("/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/test_refreezing/test_data.nc")
# ds = xr.open_dataset("/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/test_refreezing/test_data.nc")
# parameter.to_csv("/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/test_refreezing/test_parameters.csv")
# parameter = pd.read_csv("/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/test_refreezing/test_parameters.csv", index_col = 0, header= None, squeeze = True)

## Glacier melt module:

temp = ds["temp_mean"]
prec = ds["RRR"]
pdd = ds["pdd"]

reduced_temp = (parameter.TT_rain - temp) / (parameter.TT_rain - parameter.TT_snow)
snowfrac = np.clip(reduced_temp, 0, 1)
accu_rate = snowfrac * prec

# initialize snow depth and melt rates (pypdd.py line 214)
snow_depth = xr.zeros_like(temp)
snow_melt = xr.zeros_like(temp)
ice_melt = xr.zeros_like(temp)

# compute snow depth and melt rates (pypdd.py line 219)
for i in np.arange(len(temp)):
    if i > 0:
        snow_depth[i] = snow_depth[i - 1]
    snow_depth[i] += accu_rate[i]
    snow_melt[i], ice_melt[i] = MATILDA.melt_rates(snow_depth[i], pdd[i], parameter)
    snow_depth[i] -= snow_melt[i]
total_melt = snow_melt + ice_melt
refr_ice = parameter.CFR_ice * ice_melt
refr_snow = parameter.CFR_snow * snow_melt
runoff_rate = total_melt - refr_snow - refr_ice
inst_smb = accu_rate - runoff_rate

glacier_melt = xr.merge(
    [xr.DataArray(inst_smb, name="DDM_smb"),
     xr.DataArray(pdd, name="pdd"),
     xr.DataArray(accu_rate, name="DDM_accumulation_rate"),
     xr.DataArray(ice_melt, name="DDM_ice_melt"),
     xr.DataArray(snow_melt, name="DDM_snow_melt"),
     xr.DataArray(total_melt, name="DDM_total_melt"),
     xr.DataArray(refr_ice, name="DDM_refreezing_ice"),
     xr.DataArray(refr_snow, name="DDM_refreezing_snow"),
     xr.DataArray(runoff_rate, name="Q_DDM")])

# making the final dataframe
DDM_results_old = glacier_melt.to_dataframe()

## Adapted glacier melt module:
#
# parameter = parameter.append(pd.Series({'rho_snow': 400}))
#
# # Constants
# rho_ice = 917           # density of solid ice (kg/m^3)
# T_melt = parameter.TT_snow              # melting temperature of ice (°C)
# c_i = 2.1 * 10 ** -3    # specific heat of ice (MJ kg^−1 °C^−1)
# lhf = 0.334             # latent heat of fusion: 334 J/g --> 0.334 MJ/kg
# CFR_ice = 0.01          # fraction of ice melt refreezing in moulins
#
# temp = ds["temp_mean"]
# prec = ds["RRR"]
# pdd = ds["pdd"]
#
# reduced_temp = (parameter.TT_rain - temp) / (parameter.TT_rain - parameter.TT_snow)
# snowfrac = np.clip(reduced_temp, 0, 1)
# accu_rate = snowfrac * prec
#
# # initialize snow depth and melt rates (pypdd.py line 214)
# snow_depth = xr.zeros_like(temp)
# snow_melt = xr.zeros_like(temp)
# ice_melt = xr.zeros_like(temp)
#
# # calculate ice fraction in snowpack
# theta_i = parameter.rho_snow / rho_ice
#
# # calculate irreducible water content:
# if theta_i <= 0.23:
#     theta_e = 0.0264 + 0.0099 * ((1 - theta_i) / theta_i)
# elif 0.23 < theta_i <= 0.812:
#     theta_e = 0.08 - 0.1023 * (theta_i - 0.03)
# else:
#     theta_e = 0
#
# # calculate cold content (MJ m^−2):
# cc_temp = xr.where(temp < T_melt, -temp, T_melt)  # negative degree days available to refreeze melt water
#
#
# # compute snow depth and melt rates (pypdd.py line 219)
# for i in np.arange(len(temp)):
#     if i > 0:
#         snow_depth[i] = snow_depth[i - 1]
#     snow_depth[i] += accu_rate[i]
#     snow_melt[i], ice_melt[i] = MATILDA.melt_rates(snow_depth[i], pdd[i], parameter)
#     snow_depth[i] -= snow_melt[i]
# cc = cc_temp * (snow_depth / 1000) * c_i * parameter.rho_snow   # cold content in kg/m² ^= mm
# refr_pot = cc / lhf                  # refreezing potential, kg/m² ^=  mm
#
# total_melt = snow_melt + ice_melt
#
# refr_max = xr.where(snow_depth > 0, theta_e * snow_depth, 0)     # fraction of snow_depth or snow_melt??
# refreezing = np.minimum(refr_max, refr_pot)     # refreezing limited by maximum water holding capacity of the snowpack
# refreezing = np.minimum(snow_melt, refreezing)     # refreezing limited by amount of available melt water
# refr_snow = CFR_ice * ice_melt
# runoff_rate = total_melt - refreezing - refr_snow
# inst_smb = accu_rate - runoff_rate
#
# glacier_melt = xr.merge(
#     [xr.DataArray(inst_smb, name="DDM_smb"),
#      xr.DataArray(pdd, name="pdd"),
#      xr.DataArray(accu_rate, name="DDM_accumulation_rate"),
#      xr.DataArray(ice_melt, name="DDM_ice_melt"),
#      xr.DataArray(snow_melt, name="DDM_snow_melt"),
#      xr.DataArray(total_melt, name="DDM_total_melt"),
#      xr.DataArray(refr_snow, name="DDM_Refreezing_ice"),
#      xr.DataArray(refreezing, name="DDM_Refreezing_snow"),
#      xr.DataArray(runoff_rate, name="Q_DDM")])
#
# # making the final dataframe
# DDM_results_rho300 = glacier_melt.to_dataframe()

## Compare results
#
# # print(DDM_results_old.sum())
# print(DDM_results_rho300.sum())
#
# # print(DDM_results_old.describe())
# # print(DDM_results_rho300.describe())
#
# refr_max.sum()
# refr_pot.sum()
#
# refr_check = xr.merge([xr.DataArray(refr_max, name='ref_max'), xr.DataArray(refr_pot, name="ref_pot")])
# refr_check = refr_check.to_dataframe()
#



## Storage release scheme (Stahl et.al. 2008)

KG_min = 0.1        # minimum outflow coefficient (conditions with deep snow and poorly developed glacial drainage systems) [time^−1]
d_KG = 0.9          # KG_min + d_KG = maximum outflow coefficient (representing late-summer conditions with bare ice and a well developed glacial drainage system)

AG = 98        # calibration parameter (mm^−1) --> [0.1, 100000] --> high values result in KG close to 1 --> small reservoir


## Option 1:

actual_runoff = xr.zeros_like(temp)
glacier_reservoir = xr.zeros_like(temp)

KG = np.minimum(KG_min + d_KG * np.exp(snow_depth / -AG), 1)
for i in np.arange(len(temp)):
    if i == 0:
        SG = runoff_rate[i]     # liquid water stored in the reservoir
    else:
        SG = np.maximum((runoff_rate[i] - actual_runoff[i-1]) + SG, 0)
    actual_runoff[i] = KG[i] * SG
    glacier_reservoir[i] = SG


rout_check = xr.merge([xr.DataArray(actual_runoff, name='actual_runoff'),
                       xr.DataArray(runoff_rate, name="runoff_rate"),
                       xr.DataArray(glacier_reservoir, name="glacier_reservoir")])
rout_check = rout_check.to_dataframe()

rout_check.sum()

import matplotlib.pyplot as plt
rout_check[rout_check.columns[0:2]].resample('W').sum()[slice('1986-01-01', '1989-12-31')].plot()
plt.show()



##

parameter = parameter.append(pd.Series({'AG': 98}))

# initialize arrays
temp = ds["temp_mean"]
prec = ds["RRR"]
pdd = ds["pdd"]
snow_depth = xr.zeros_like(temp)
snow_melt = xr.zeros_like(temp)
ice_melt = xr.zeros_like(temp)
actual_runoff = xr.zeros_like(temp)
glacier_reservoir = xr.zeros_like(temp)

# calculate accumulation (SWE)
reduced_temp = (parameter.TT_rain - temp) / (parameter.TT_rain - parameter.TT_snow)
snowfrac = np.clip(reduced_temp, 0, 1)
accu_rate = snowfrac * prec

# compute snow depth and melt rates
for i in np.arange(len(temp)):
    if i > 0:
        snow_depth[i] = snow_depth[i - 1]
    snow_depth[i] += accu_rate[i]
    snow_melt[i], ice_melt[i] = MATILDA.melt_rates(snow_depth[i], pdd[i], parameter)
    snow_depth[i] -= snow_melt[i]

# calculate refreezing, runoff and surface mass balance
total_melt = snow_melt + ice_melt
refr_ice = parameter.CFR_ice * ice_melt
refr_snow = parameter.CFR_snow * snow_melt
runoff_rate = total_melt - refr_snow - refr_ice
inst_smb = accu_rate - runoff_rate

# Storage-release scheme for glacier outflow (Stahl et.al. 2008, Toum et. al. 2021)
KG_min = 0.1        # minimum outflow coefficient (conditions with deep snow and poorly developed glacial drainage systems) [time^−1]
d_KG = 0.9          # KG_min + d_KG = maximum outflow coefficient (representing late-summer conditions with bare ice and a well developed glacial drainage system) [time^−1]
KG = np.minimum(KG_min + d_KG * np.exp(snow_depth / -parameter.AG), 1)
for i in np.arange(len(temp)):
    if i == 0:
        SG = runoff_rate[i]     # liquid water stored in the reservoir
    else:
        SG = np.maximum((runoff_rate[i] - actual_runoff[i-1]) + SG, 0)
    actual_runoff[i] = KG[i] * SG
    glacier_reservoir[i] = SG


# final glacier module output
glacier_melt = xr.merge(
    [xr.DataArray(inst_smb, name="DDM_smb"),
     xr.DataArray(pdd, name="pdd"),
     xr.DataArray(accu_rate, name="DDM_accumulation_rate"),
     xr.DataArray(ice_melt, name="DDM_ice_melt"),
     xr.DataArray(snow_melt, name="DDM_snow_melt"),
     xr.DataArray(total_melt, name="DDM_total_melt"),
     xr.DataArray(refr_ice, name="DDM_refreezing_ice"),
     xr.DataArray(refr_snow, name="DDM_refreezing_snow"),
     xr.DataArray(glacier_reservoir, name="glacier_reservoir"),
     xr.DataArray(actual_runoff, name='Q_DDM')
     ])

DDM_results = glacier_melt.to_dataframe()

DDM_results.sum()



