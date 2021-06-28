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
input_df = home + "/Seafile/SHK/Scripts/centralasiawaterresources/Test_area/Test_BA_CMIP-4-5/model_output_2021-2100.csv"
glacier_profile = home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/BA-glacier_profile.txt"
parameter = home + "/Seafile/SHK/Scripts/centralasiawaterresources/Test_area/Test_BA_CMIP-4-5/model_parameter.csv"
area_cat = 46.23
area_glac = 2.566
hydro_year = 10

df = pd.read_csv(input_df)
glacier_profile = pd.read_csv(glacier_profile)
parameter = pd.read_csv(parameter)

test_df = df[['TIMESTAMP', 'DDM_smb', 'Q_DDM']].copy().set_index("TIMESTAMP")
test_df.index = pd.to_datetime(test_df.index)

## lookup table
def create_lookup_table(glacier_profile, area_cat):
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

lookup_table = create_lookup_table(glacier_profile, area_cat)

##
test_df["water_year"] = np.where((test_df.index.month) >= hydro_year, test_df.index.year + 1,
                                            test_df.index.year)
test_df["Q_DDM_updated"] = test_df["Q_DDM"].copy()
# total water equivalent of the glacier in mm w.
m = sum((glacier_profile["Area"] * area_cat) * glacier_profile["WE"])
initial_area = glacier_profile.groupby("EleZone")["Area"].sum()

glacier_change = pd.DataFrame({"smb": test_df.groupby("water_year")["DDM_smb"].sum() * 0.9}).reset_index()  # do we have to scale this?
glacier_change["smb_sum"] = np.cumsum(glacier_change["smb"])
# percentage of how much of the initial mass melted
glacier_change["smb_percentage"] = round((glacier_change["smb_sum"] / m) * 100)

glacier_change_area = pd.DataFrame({"time":"initial", "glacier_area":[area_glac]})

for i in range(len(glacier_change)):
    year = glacier_change["water_year"][i]
    smb_sum = glacier_change["smb_sum"][i]
    smb = int(-glacier_change["smb_percentage"][i])
    if smb <=99:
        # getting the right row from the lookup table depending on the smb
        area_melt = lookup_table.iloc[smb]
        # getting the new glacier area by multiplying the initial area with the area changes
        new_area = np.nansum((area_melt.values * (initial_area.values)))*area_cat
    else:
        new_area = 0
    # multiplying the output with the fraction of the new area
    glacier_change_area = glacier_change_area.append({'time': year, "glacier_area":new_area, "smb_sum":smb_sum}, ignore_index=True)
    test_df["Q_DDM_updated"] = np.where(test_df["water_year"] == year, test_df["Q_DDM"] * (new_area / area_cat), test_df["Q_DDM_updated"])


## my way, probably false
# initial smb from the glacier routine script in m w.e.
m = sum((glacier_profile["Area"] * area_cat) * glacier_profile["WE"])
initial_smb = m / 1000 # in m
# initial area
initial_area = glacier_profile.groupby("EleZone")["Area"].sum()
# dataframe with the smb change per hydrological year in m w.e.
glacier_change = pd.DataFrame({"smb": test_df.groupby("water_year")["DDM_smb"].sum() / 1000 * 0.9}).reset_index()  # do we have to scale this?
# adding the changes to get the whole change in comparison to the initial area
glacier_change["smb_sum"] = np.cumsum(glacier_change["smb"])
# percentage of how much of the initial mass melted
glacier_change["smb_percentage"] = round((glacier_change["smb_sum"] / initial_smb) * 100)

glacier_change_area = pd.DataFrame({"time":"initial", "glacier_area":[area_glac]})

test_df["Q_DDM_updated"] = test_df["Q_DDM"].copy()
for i in range(len(glacier_change)):
    year = glacier_change["water_year"][i]
    smb_sum = glacier_change["smb_sum"][i]
    smb = int(-glacier_change["smb_percentage"][i])
    if smb <=99:
        # getting the right row from the lookup table depending on the smb
        area_melt = lookup_table.iloc[smb]
        # getting the new glacier area by multiplying the initial area with the area changes
        new_area = np.nansum((area_melt.values * (initial_area.values)))*area_cat
    else:
        new_area = 0
    # multiplying the output with the fraction of the new area
    glacier_change_area = glacier_change_area.append({'time': year, "glacier_area":new_area, "smb_sum":smb_sum}, ignore_index=True)
    test_df["Q_DDM_updated"] = np.where(test_df["water_year"] == year, test_df["Q_DDM"] * (new_area / area_cat), test_df["Q_DDM_updated"])

## test HBV Light

df_hbv = df[['TIMESTAMP', 'RRR', 'T2']].copy()
df_hbv.columns = ["Date", "P", "T"]
df_hbv["Q"] = 0
df_hbv["Date"] = pd.to_datetime(df_hbv["Date"])
df_hbv["Date"] = df_hbv["Date"].apply(lambda x: x.strftime('%Y%m%d'))
df_hbv2 = df[['TIMESTAMP', 'PE']].copy()

##
df = df.set_index("TIMESTAMP")
df.index = pd.to_datetime(df.index)

hbv_light = pd.read_csv(home + "/Seafile/Ana-Lena_Phillip/data/HBV-Light/HBV-light_data/Bash_Kaindy/Test_CMIP4-5/Results/Results.txt", sep="\t")
hbv_light["Date"] = hbv_light["Date"].apply(lambda x: pd.to_datetime(str(x), format='%Y%m%d'))
hbv_light = hbv_light.set_index("Date")
hbv_light.index = pd.to_datetime(hbv_light.index)

comparison = df.merge(hbv_light, left_index=True, right_index=True)
comparison_yearly = comparison.resample("Y").agg(
    {"Q_Total":"sum", "Qsim":"sum"})

plt.plot(comparison_yearly.index.to_pydatetime(), comparison_yearly["Q_Total"], label="MATILDA")
plt.plot(comparison_yearly.index.to_pydatetime(), comparison_yearly["Qsim"], label="HBV Light")
plt.legend()
plt.title("CMIP 4.5 run Bash Kaindy")
plt.show()


