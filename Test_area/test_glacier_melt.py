## Running all the required functions
from pathlib import Path; home = str(Path.home())
import pandas as pd
import numpy as np
from datetime import datetime
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA_slim import MATILDA

##
input_df = home + "/Seafile/Tianshan_data/CMIP/CMIP6/all_models/Bash_Kaindy/CMIP6_mean_41-75.9_1980-01-01-2100-12-31_downscaled.csv"
glacier_profile = home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/BA-glacier_profile.txt"
parameter = home + "/Seafile/SHK/Scripts/centralasiawaterresources/Test_area/Test_BA_CMIP4-5_old-para/model_parameter.csv"

df = pd.read_csv(input_df)
glacier_profile = pd.read_csv(glacier_profile)
parameter = pd.read_csv(parameter).set_index("Unnamed: 0")

df = df[["time","temp_45", "prec_45"]]
df.columns = ['TIMESTAMP', 'T2', 'RRR']
df.set_index('TIMESTAMP', inplace=True)
df.index = pd.to_datetime(df.index)
df = df['2021-01-01 12:00:00': '2040-12-31 12:00:00']

parameter = MATILDA.MATILDA_parameter(df, set_up_start='2021-01-01 12:00:00', set_up_end='2021-12-31 12:00:00',
                                      sim_start='2022-01-01 12:00:00', sim_end='2040-12-31 12:00:00', freq="D",
                                      lat=41, area_cat=46.23, area_glac=2.566, ele_dat=2250, ele_glac=4035, ele_cat=3485,
                                      CFMAX_ice=5, CFMAX_snow=2.5, BETA=1, CET=0.15, FC=200, K0=0.055, K1= 0.055, K2=0.04,
                                      LP=0.7, MAXBAS=2, PERC=2.5, UZL=60, TT_snow=-0.5, TT_rain=2, SFCF=0.7, CFR_ice=0.05,
                                      CFR_snow= 0.05, CWH=0.1)
df_preproc = MATILDA.MATILDA_preproc(df, parameter)
output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, glacier_profile=glacier_profile)

# Scaling the temperature to the glacier mean elevation
df["T2_glac"] = (df["T2"] + (float(parameter.loc["ele_glac"].values.item())-float(parameter.loc["ele_dat"].values.item())) * float(-0.006)) - 273.15
df["T2_glac"].mean()

# calculation the postive degree days (sum of the temperatures over 0 °C)
df["pdd"] = np.where(df["T2_glac"] > 0, df["T2_glac"], 0)

## Calculation glacier melt and the SMB with pypdd
# pdd uses yearly values and everything in m
# we changed the unit to mm, so precipitation is in mm and the degree day factors too
# we use daily values

# temp is computed with the stdv and an integral formulation (Calov and Greve, 2005)
# pdd are then multiplied by 365 to get a total sum including all days of the year
temp = xr.DataArray(df["T2_glac"].copy())
prec = xr.DataArray(df["RRR"].copy())
pdd = xr.DataArray(df["pdd"].copy())

# The fraction of precipitation that falls as snow decreases linearly from one to zero between temperature thresholds
# defined by the"TT_snow" and "TT_rain" attributes
reduced_temp = (float(parameter.loc["TT_rain"].values.item()) - temp) / (float(parameter.loc["TT_rain"].values.item()) - float(parameter.loc["TT_snow"].values.item()))
# if temperature is under threshold or within, the fraction is set to 1 to indicate that snow falls
snowfrac = np.clip(reduced_temp, 0, 1) # this should be correct, I tested it manually in a crude way
# accumulation rate is the snow fraction (does prec fall as snow?) times precipitation
accu_rate = snowfrac * prec

snow_depth = xr.zeros_like(temp)
snow_melt_rate = xr.zeros_like(temp)
ice_melt_rate = xr.zeros_like(temp)

def melt_rates(snow, pdd):
    # compute a potential snow melt
    pot_snow_melt = float(parameter.loc["CFMAX_snow"].values.item()) * pdd
    # effective snow melt can't exceed amount of snow
    snow_melt = np.minimum(snow, pot_snow_melt)
    # ice melt is proportional to excess snow melt
    ice_melt = (pot_snow_melt - snow_melt) * (float(parameter.loc["CFMAX_ice"].values.item()) / float(parameter.loc["CFMAX_snow"].values.item()))
    return (snow_melt, ice_melt)

# compute snow depth and melt rates (pypdd.py line 219)
# the accumulation rate is added to the snow depth after each day
# potential snow melt is calculated by multiplying pdd with the ddf for snow
# actual snowmelt is calculated by checking how much snow and how much snow melt there is. The minimum is taken as the snow
# melt can't exceed the snow
# additional melt energy is used for ice melt
# snow melt is removed from the snow depth for each day

for i in range(len(temp)):
    if i > 0:
        snow_depth[i] = snow_depth[i - 1]
    snow_depth[i] += accu_rate[i]
    snow_melt_rate[i], ice_melt_rate[i] = melt_rates(snow_depth[i], pdd[i])
    snow_depth[i] -= snow_melt_rate[i]
# all melt
total_melt = snow_melt_rate + ice_melt_rate
# runoff is all melt minus the refreezing of snow and ice
# precipitation is NOT included!
runoff_rate = total_melt - float(parameter.loc["CFR_snow"].values.item()) * snow_melt_rate - float(parameter.loc["CFR_ice"].values.item()) * ice_melt_rate
# mass balance is the accumulation minus the runoff
inst_smb = accu_rate - runoff_rate

glacier_melt = xr.merge(
    [xr.DataArray(inst_smb, name="DDM_smb"), xr.DataArray(pdd, name="pdd"), \
     xr.DataArray(accu_rate, name="DDM_accumulation_rate"),
     xr.DataArray(ice_melt_rate, name="DDM_ice_melt_rate"),
     xr.DataArray(snow_depth, name="DDM_snow_depth"),
     xr.DataArray(snow_melt_rate, name="DDM_snow_melt_rate"), \
     xr.DataArray(total_melt, name="DDM_total_melt"), xr.DataArray(runoff_rate, name="Q_DDM")])

# making the final dataframe
DDM_results = glacier_melt.to_dataframe()
DDM_results = DDM_results.round(3)

# different from the HBV model where the melted snow is added to the snowpack or glacier and refreezes


## lookup table: validated by Marc Vis and HBV-Light returns the same table
def create_lookup_table(glacier_profile, area_glac):
    initial_area = glacier_profile["Area"]  # per elevation band
    hi_initial = glacier_profile["WE"]  # initial water equivalent of each elevation band
    hi_k = glacier_profile[
        "WE"]  # hi_k is the updated water equivalent for each elevation zone, starts with initial values
    ai = glacier_profile["Area"]  # ai is the glacier area of each elevation zone, starts with initial values

    lookup_table = pd.DataFrame()
    lookup_table = lookup_table.append(initial_area, ignore_index=True)

    # Pre-simulation
    # 1. calculate total glacier mass in mm water equivalent: M = sum(ai * hi)
    m = sum(glacier_profile["Area"] * glacier_profile["WE"])

    # melt the glacier in steps of 1 percent
    deltaM = -m / 100

    # 2. Normalize glacier elevations: Einorm = (Emax-Ei)/(Emax-Emin)
    glacier_profile["norm_elevation"] = (glacier_profile["Elevation"].max() - glacier_profile["Elevation"]) / \
                                        (glacier_profile["Elevation"].max() - glacier_profile["Elevation"].min())
    # 3. Apply deltaH parameterization: deltahi = (Einorm+a)^y + b*(Einorm+a) + c
    # deltahi is the normalized (dimensionless) ice thickness change of elevation band i
    # choose one of the three parameterizations from Huss et al. (2010) depending on glacier size
    if area_glac < 5:
        a = -0.3
        b = 0.6
        c = 0.09
        y = 2
    elif area_glac < 20:
        a = -0.05
        b = 0.19
        c = 0.01
        y = 4
    else:
        a = -0.02
        b = 0.12
        c = 0
        y = 6

    glacier_profile["delta_h"] = (glacier_profile["norm_elevation"] + a) ** y + (
            b * (glacier_profile["norm_elevation"] + a)) + c

    ai_scaled = ai.copy()  # setting ai_scaled with the initial values

    fs = deltaM / (sum(ai * glacier_profile["delta_h"]))  # a) initial ai

    for _ in range(99):
        # 5. compute glacier geometry for reduced mass
        hi_k = hi_k + fs * glacier_profile["delta_h"]

        leftover = sum(
            pd.Series(np.where(hi_k < 0, hi_k, 0)) * ai)  # Calculate leftover (i.e. the 'negative' glacier volume)

        hi_k = pd.Series(np.where(hi_k < 0, np.nan,
                                  hi_k))  # Set those zones that have a negative weq to NaN to make sure they will be excluded from now on

        # 6. width scaling
        ai_scaled = ai * np.sqrt((hi_k / hi_initial))

        # 7. create lookup table
        # glacier area for each elevation band for 101 different mass situations (100 percent to 0 in 1 percent steps)

        lookup_table = lookup_table.append(ai_scaled, ignore_index=True)

        if sum(pd.Series(np.where(np.isnan(ai_scaled), 0, ai)) * glacier_profile["delta_h"]) == 0:
            ai_scaled = np.where(ai_scaled == 1, 1, 0)
        else:
            # Update fs (taking into account the leftover)
            fs = (deltaM + leftover) / sum(
                pd.Series(np.where(np.isnan(ai_scaled), 0, ai)) * glacier_profile["delta_h"])

    lookup_table = lookup_table.fillna(0)

    lookup_table.columns = glacier_profile["EleZone"]
    lookup_table = lookup_table.groupby(level=0, axis=1).sum()

    elezones_inital = lookup_table.iloc[0]

    lookup_table = lookup_table / elezones_inital
    lookup_table = round(lookup_table, 4)
    lookup_table.iloc[-1] = 0
    return lookup_table

lookup_table = create_lookup_table(glacier_profile, float(parameter.loc["area_glac"].values.item()))

##
hydro_year = 10
test_df = DDM_results.copy()

test_df["water_year"] = np.where((test_df.index.month) >= hydro_year, test_df.index.year + 1,
                                            test_df.index.year)
test_df["Q_DDM_updated"] = test_df["Q_DDM"].copy()
# total water equivalent of the glacier in mm w.
# IS M CORRECT?
m = sum((glacier_profile["Area"]) * glacier_profile["WE"])
initial_area = glacier_profile.groupby("EleZone")["Area"].sum()
test_df["DDM_smb_scal"] = - (test_df["DDM_ice_melt_rate"] - ((test_df["DDM_ice_melt_rate"] * float(parameter.loc["CFR_ice"].values.item()))))
glacier_change = pd.DataFrame({"smb": test_df.groupby("water_year")["DDM_smb_scal"].sum() * 0.9}).reset_index()  # do we have to scale this?
glacier_change["smb_sum"] = np.cumsum(glacier_change["smb"])
# percentage of how much of the initial mass melted
glacier_change["smb_percentage"] = round((glacier_change["smb_sum"] / m) * 100)

glacier_change_area = pd.DataFrame({"time":"initial", "glacier_area":[float(parameter.loc["area_glac"].values.item())]})

for i in range(len(glacier_change)):
    year = glacier_change["water_year"][i]
    smb_sum = glacier_change["smb_sum"][i]
    smb = int(-glacier_change["smb_percentage"][i])
    if (smb <=99) & (smb >= 0):
        # getting the right row from the lookup table depending on the smb
        area_melt = lookup_table.iloc[smb]
        # getting the new glacier area by multiplying the initial area with the area changes
        new_area = np.nansum((area_melt.values * (initial_area.values)))*float(parameter.loc["area_cat"].values.item())
    else:
        new_area = 0
    # multiplying the output with the fraction of the new area
    glacier_change_area = glacier_change_area.append({'time': year, "glacier_area":new_area, "smb_sum":smb_sum}, ignore_index=True)
    #test_df["Q_DDM_updated"] = np.where(test_df["water_year"] == year, test_df["Q_DDM"] * (new_area / area_cat), test_df["Q_DDM_updated"])


## test HBV Light
df["T2_cat"] = (df["T2"] + (float(parameter.loc["ele_cat"].values.item())-float(parameter.loc["ele_dat"].values.item())) * float(-0.006)) - 273.15
doy = np.array(df.index.strftime('%j')).astype(int)
lat = np.deg2rad(41)
# Part 2. Extraterrrestrial radiation calculation
# set solar constant (in W m-2)
Rsc = 1367  # solar constant (in W m-2)
# calculate solar declination dt (in radians)
dt = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
# calculate sunset hour angle (in radians)
ws = np.arccos(-np.tan(lat) * np.tan(dt))
# Calculate sunshine duration N (in hours)
N = 24 / np.pi * ws
# Calculate day angle j (in radians)
j = 2 * np.pi / 365.25 * doy
# Calculate relative distance to sun
dr = 1.0 + 0.03344 * np.cos(j - 0.048869)
# Calculate extraterrestrial radiation (J m-2 day-1)
Re = Rsc * 86400 / np.pi * dr * (ws * np.sin(lat) * np.sin(dt) \
                                 + np.sin(ws) * np.cos(lat) * np.cos(dt))
# convert from J m-2 day-1 to MJ m-2 day-1
Re = Re / 10 ** 6

df["PE"] = np.where(df["T2_cat"] + 5 > 0, (Re / (water_density * latent_heat_flux)) * ((df["T2_cat"] + 5) / 100) * 1000, 0)
##
df["TIMESTAMP"] = df.index
df_hbv = df[['TIMESTAMP', 'RRR', 'T2_cat']].copy()
df_hbv.columns = ["Date", "P", "T"]
df_hbv["Q"] = 0
df_hbv["Date"] = pd.to_datetime(df_hbv["Date"])
df_hbv["Date"] = df_hbv["Date"].apply(lambda x: x.strftime('%Y%m%d'))
df_hbv2 = df[['PE']].copy()


##

hbv_light = pd.read_csv(home + "/Seafile/Ana-Lena_Phillip/data/HBV-Light/HBV-light_data/Bash_Kaindy/Test_CMIP4-5_40years/Results/Results.txt", sep="\t")
hbv_light["Date"] = hbv_light["Date"].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
hbv_light = hbv_light.set_index("Date")
hbv_light.index = pd.to_datetime(hbv_light.index)

#output_MATILDA[0]["Q_Total"] = df["Q_HBV"] +test_df["Q_DDM_updated"]

comparison = output_MATILDA[0].merge(hbv_light, left_index=True, right_index=True)
comparison_yearly = comparison.resample("Y").agg(
    {"Q_Total":"sum", "Qsim":"sum"})

plt.plot(comparison_yearly.index.to_pydatetime(), comparison_yearly["Q_Total"], label="MATILDA")
plt.plot(comparison_yearly.index.to_pydatetime(), comparison_yearly["Qsim"], label="HBV Light")
plt.legend()
plt.title("CMIP 4.5 run Bash Kaindy")
plt.show()

