# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
# Import all necessary python packages
import sys

import numpy
import xarray as xr
import numpy as np
import pandas as pd
import scipy.signal as ss
import copy
import hydroeval
import HydroErr as he
import warnings
warnings.filterwarnings(action='ignore' ,module='HydroErr')
from datetime import date, datetime, timedelta
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText

from plotly.subplots import make_subplots
import plotly.graph_objects as go

# Setting the parameter for the MATILDA simulation
def matilda_parameter(input_df, set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D",
                      lat= None, area_cat=None, area_glac=None, ele_dat=None, ele_glac=None, ele_cat=None, parameter_set = None,
                      soi = None, warn = False, pfilter=0.2,
                      lr_temp=-0.006, lr_prec=0, \
                      hydro_year=10, TT_snow=0, TT_diff=2, CFMAX_snow=2.8, CFMAX_rel=2, \
                      BETA=1.0, CET=0.15, FC=250, K0=0.055, K1=0.055, K2=0.04, LP=0.7, MAXBAS=3.0, \
                      PERC=1.5, UZL=120, PCORR=1.0, SFCF=0.7, CWH=0.1, AG=0.7, RFS=0.15,
                      # Constants
                      CFR_ice = 0.01,              # fraction of ice melt refreezing in moulins
                        **kwargs):

    """Creates a series from the provided and/or default parameters to be provided to all subsequent MATILDA modules."""

    # Filter warnings:
    if not warn:
        warnings.filterwarnings(action='ignore')

    # takes parameters directly from a dataframe, e.g. the output from SPOTPY
    if parameter_set is not None:
        if isinstance(parameter_set, dict):
            parameter_set = pd.DataFrame(parameter_set, index=[0]).transpose()
        elif isinstance(parameter_set, pd.DataFrame):
            parameter_set = parameter_set.set_index(parameter_set.columns[0])
        else:
            print("ERROR: parameter_set can either be passed as dict and or pd.DataFrames!")
            return

        if "lr_temp" in parameter_set.index:
            lr_temp = parameter_set.loc["lr_temp"].values.item()
        if "lr_prec" in parameter_set.index:
            lr_prec = parameter_set.loc["lr_prec"].values.item()
        if "BETA" in parameter_set.index:
            BETA = parameter_set.loc["BETA"].values.item()
        if "CET" in parameter_set.index:
            CET = parameter_set.loc["CET"].values.item()
        if "FC" in parameter_set.index:
            FC = parameter_set.loc["FC"].values.item()
        if "K0" in parameter_set.index:
            K0 = parameter_set.loc["K0"].values.item()
        if "K1" in parameter_set.index:
            K1 = parameter_set.loc["K1"].values.item()
        if "K2" in parameter_set.index:
            K2 = parameter_set.loc["K2"].values.item()
        if "LP" in parameter_set.index:
            LP = parameter_set.loc["LP"].values.item()
        if "MAXBAS" in parameter_set.index:
            MAXBAS = parameter_set.loc["MAXBAS"].values.item()
        if "PERC" in parameter_set.index:
            PERC = parameter_set.loc["PERC"].values.item()
        if "UZL" in parameter_set.index:
            UZL = parameter_set.loc["UZL"].values.item()
        if "PCORR" in parameter_set.index:
            PCORR = parameter_set.loc["PCORR"].values.item()
        if "TT_snow" in parameter_set.index:
            TT_snow = parameter_set.loc["TT_snow"].values.item()
        if "TT_diff" in parameter_set.index:
            TT_diff = parameter_set.loc["TT_diff"].values.item()
        if "CFMAX_snow" in parameter_set.index:
            CFMAX_snow = parameter_set.loc["CFMAX_snow"].values.item()
        if "CFMAX_rel" in parameter_set.index:
            CFMAX_rel = parameter_set.loc["CFMAX_rel"].values.item()
        if "SFCF" in parameter_set.index:
            SFCF = parameter_set.loc["SFCF"].values.item()
        if "CFR_ice" in parameter_set.index:
            CFR_ice = parameter_set.loc["CFR_ice"].values.item()
        if "CWH" in parameter_set.index:
            CWH = parameter_set.loc["CWH"].values.item()
        if "AG" in parameter_set.index:
            AG = parameter_set.loc["AG"].values.item()
        if "RFS" in parameter_set.index:
            RFS = parameter_set.loc["RFS"].values.item()

    print("Reading parameters for MATILDA simulation")
    # Checking the parameters to set the catchment properties and simulation
    if lat is None:
        print("WARNING: No latitude specified. Please provide to calculate PE")
        return
    if area_cat is None:
        print("WARNING: No catchment area specified. Please provide catchment area in km2")
        return
    if area_glac is None:
        area_glac = 0
    if area_glac > area_cat:
        print("ERROR: Glacier area exceeds overall catchment area")
        return
    if ele_dat is not None and ele_cat is None:
        print("WARNING: Catchment reference elevation is missing. The data can not be elevation scaled.")
    if ele_cat is None or ele_glac is None:
        print("WARNING: Reference elevations for catchment and glacier area need to be provided to scale the model"
              "domains correctly!")
        ele_non_glac = None
    else:
        # Calculate the mean elevation of the non-glacierized catchment area
        ele_non_glac = (ele_cat - area_glac / area_cat * ele_glac) * area_cat / (area_cat - area_glac)
    if area_glac is not None or area_glac > 0:
        if ele_glac is None and ele_dat is not None:
            print("WARNING: Glacier reference elevation is missing")
    if hydro_year > 12 and hydro_year < 1:
        print("WARNING: Beginning of hydrological year out of bounds [1, 12]")

    if set_up_end is not None and sim_start is not None:
        if set_up_end > sim_start:
            print("WARNING: Set up period overlaps start of simulation period")
    if set_up_start is None and sim_start is None:
        set_up_start = input_df["TIMESTAMP"].iloc[0]
    if set_up_end is None and sim_end is None:
        set_up_end = pd.to_datetime(input_df["TIMESTAMP"].iloc[0])
        set_up_end = set_up_end + pd.DateOffset(years=1)
        set_up_end = str(set_up_end)
    if sim_start is None:
        sim_start = input_df["TIMESTAMP"].iloc[0]
    if sim_end is None:
        sim_end = input_df["TIMESTAMP"].iloc[-1]
    if set_up_start is None and sim_start is not None:
        if sim_start == input_df["TIMESTAMP"].iloc[0]:
            set_up_start = sim_start
        else:
            set_up_start = pd.to_datetime(sim_start) + pd.DateOffset(years=-1)
        set_up_end = pd.to_datetime(set_up_start) + pd.DateOffset(years=1) + pd.DateOffset(days=-1)
        set_up_start = str(set_up_start)
        set_up_end = str(set_up_end)

    freq_long = ""
    if freq == "D":
        freq_long = "Daily"
    elif freq == "W":
        freq_long = "Weekly"
    elif freq == "M":
        freq_long = "Monthly"
    elif freq == "Y":
        freq_long = "Annual"
    else:
        print(
            "WARNING: Resampling rate " + freq + " is not supported. Choose either 'D' (daily), 'W' (weekly), 'M' (monthly) or 'Y' (yearly).")

    # Check if season of interest is specified
    if soi is not None:
        if type(soi) is not list:
            print("Error: The season of interest (soi) needs to be specified as 2-element list: [first_calendar_month, last_calendar_month]")
            sys.exit()
        elif len(soi) is not 2:
            print("Error: The season of interest (soi) needs to be specified as 2-element list: [first_calendar_month, last_calendar_month]")
            sys.exit()


    # Check model parameters
    if 0 > pfilter or lr_temp > 0.5:
        print("WARNING: Parameter pfilter exceeds the recommended threshold [0, 0.5].")
    if -0.01 > lr_temp or lr_temp > -0.003:
        print("WARNING: Parameter lr_temp exceeds boundaries [-0.01, -0.003].")
    if 0 > lr_prec or lr_prec > 0.002:
        print("WARNING: Parameter lr_prec exceeds boundaries [0, 0.002].")
    if 1 > BETA or BETA > 6:
        print("WARNING: Parameter BETA exceeds boundaries [1, 6].")
    if 0 > CET or CET > 0.3:
        print("WARNING: Parameter CET exceeds boundaries [0, 0.3].")
    if 50 > FC or FC > 500:
        print("WARNING: Parameter FC exceeds boundaries [50, 500].")
    if 0.01 > K0 or K0 > 0.4:
        print("WARNING: Parameter K0 exceeds boundaries [0.01, 0.4].")
    if 0.01 > K1 or K1 > 0.4:
        print("WARNING: Parameter K1 exceeds boundaries [0.01, 0.4].")
    if 0.001 > K2 or K2 > 0.15:
        print("WARNING: Parameter K2 exceeds boundaries [0.001, 0.15].")
    if 0.3 > LP or LP > 1:
        print("WARNING: Parameter LP exceeds boundaries [0.3, 1].")
    if 1 >= MAXBAS or MAXBAS > 7:
        print("ERROR: Parameter MAXBAS exceeds boundaries [2, 7]. Please choose a suitable value.")
        return
    if 0 > PERC or PERC > 3:
        print("WARNING: Parameter PERC exceeds boundaries [0, 3].")
    if 0 > UZL or UZL > 500:
        print("WARNING: Parameter UZL exceeds boundaries [0, 500].")
    if 0.5 > PCORR or PCORR > 2:
        print("WARNING: Parameter PCORR exceeds boundaries [0.5, 2].")
    if -1.5 > TT_snow or TT_snow > 2.5:
        print("WARNING: Parameter TT_snow exceeds boundaries [-1.5, 2.5].")
    if 0.2 > TT_diff or TT_diff > 4:
        print("WARNING: Parameter TT_diff exceeds boundaries [0.2, 4].")
    if 1.2 > CFMAX_rel or CFMAX_rel > 2.5:
        print("WARNING: Parameter CFMAX_rel exceeds boundaries [1.2, 2.5].")
    if 1 > CFMAX_snow or CFMAX_snow > 10:
        print("WARNING: Parameter CFMAX_snow exceeds boundaries [1, 10].")
    if 0.4 > SFCF or SFCF > 1:
        print("WARNING: Parameter SFCF exceeds boundaries [0.4, 1].")
    if 0 > CWH or CWH > 0.2:
        print("WARNING: Parameter CWH exceeds boundaries [0, 0.2].")
    if 0 > AG or AG > 1:
        print("WARNING: Parameter AG exceeds boundaries [0, 1].")
    if 0.05 > RFS or RFS > 0.25:
        print("WARNING: Parameter RFS exceeds boundaries [0.05, 0.25].")

    # calculate threshold temperature for rain
    TT_rain = TT_diff + TT_snow

    # calculate ice melt factor:
    CFMAX_ice = CFMAX_rel * CFMAX_snow

    parameter = pd.Series(
        {"set_up_start": set_up_start, "set_up_end": set_up_end, "sim_start": sim_start, "sim_end": sim_end,
         "freq": freq, "freq_long": freq_long, "lat": lat, "area_cat": area_cat, "area_glac": area_glac,
         "ele_dat": ele_dat, "ele_cat": ele_cat, "ele_glac": ele_glac, "ele_non_glac": ele_non_glac, "hydro_year": hydro_year,
         "soi": soi, "warn": warn, "pfilter": pfilter, "lr_temp": lr_temp, "lr_prec": lr_prec, "TT_snow": TT_snow,
         "TT_rain": TT_rain, "TT_diff": TT_diff, "CFMAX_snow": CFMAX_snow, "CFMAX_ice": CFMAX_ice,
         "CFMAX_rel": CFMAX_rel, "BETA": BETA, "CET": CET, "FC": FC, "K0": K0, "K1": K1, "K2": K2, "LP": LP,
         "MAXBAS": MAXBAS, "PERC": PERC, "UZL": UZL, "PCORR": PCORR, "SFCF": SFCF, "CWH": CWH, "AG": AG,
         "CFR_ice": CFR_ice, "RFS": RFS})
    print("Parameter set:")
    print(str(parameter))
    return parameter


def matilda_preproc(input_df, parameter, obs=None):
    """MATILDA preprocessing: transforms dataframes into the required format, converts observation units, and applies
    precipitation correction factor."""

    print('*-------------------*')
    print("Reading data")
    print("Set up period from " + str(parameter.set_up_start) + " to " + str(parameter.set_up_end) + " to set initial values")
    print("Simulation period from " + str(parameter.sim_start) + " to " + str(parameter.sim_end))
    df_preproc = input_df.copy()

    if parameter.set_up_start > parameter.sim_start:
        print("WARNING: Spin up period starts after simulation period")
    elif isinstance(df_preproc, xr.Dataset):
        df_preproc = input_df.sel(time=slice(parameter.set_up_start, parameter.sim_end))
    else:
        df_preproc.set_index('TIMESTAMP', inplace=True)
        df_preproc.index = pd.to_datetime(df_preproc.index)
        df_preproc = df_preproc[parameter.set_up_start: parameter.sim_end]

    # make sure temperatures are in Celsius
    df_preproc["T2"] = np.where(df_preproc["T2"] >= 100, df_preproc["T2"] - 273.15, df_preproc["T2"])

    # overall precipitation correction factor
    df_preproc['RRR'] = df_preproc['RRR'] * parameter.PCORR
    df_preproc['RRR'] = np.where(df_preproc['RRR'] < 0, 0, df_preproc['RRR'])

    if obs is not None:
        obs_preproc = obs.copy()
        obs_preproc.set_index('Date', inplace=True)
        obs_preproc.index = pd.to_datetime(obs_preproc.index)
        obs_preproc = obs_preproc[parameter.sim_start: parameter.sim_end]
        # Changing the input unit from m^3/s to mm/d.
        obs_preproc["Qobs"] = obs_preproc["Qobs"] * 86400 / (parameter.area_cat * 1000000) * 1000
        obs_preproc = obs_preproc.resample("D").agg(pd.Series.sum, skipna=False)
        # Omit everything outside the specified season of interest (soi)
        if parameter.soi is not None:
            obs_preproc = obs_preproc[obs_preproc.index.month.isin(range(parameter.soi[0], parameter.soi[1] + 1))]
        # Expanding the observation period to full years filling up with NAs
        idx_first = obs_preproc.index.year[1]
        idx_last = obs_preproc.index.year[-1]
        idx = pd.date_range(start=date(idx_first, 1, 1), end=date(idx_last, 12, 31), freq='D', name=obs_preproc.index.name)
        obs_preproc = obs_preproc.reindex(idx)
        obs_preproc = obs_preproc.fillna(np.NaN)


    if obs is not None:
        return df_preproc, obs_preproc
    if obs is None:
        return df_preproc


def phase_separation(df_preproc, parameter):
    """Separates precipitation in liquid and solid fractions with linear transition between threshold temperatures."""
    reduced_temp = (parameter.TT_rain - df_preproc['T2']) / (parameter.TT_rain - parameter.TT_snow)
    snowfrac = np.clip(reduced_temp, 0, 1)
    snow = snowfrac * df_preproc['RRR']
    rain = df_preproc['RRR'] - snow
    
    return rain, snow



def input_scaling(df_preproc, parameter):
    """Scales the input data to respective mean elevations. Separates precipitation in phases and
    applies the snow fall correction factor."""

    if parameter.ele_glac is not None:
        elev_diff_glacier = parameter.ele_glac - parameter.ele_dat
        input_df_glacier = df_preproc.copy()
        input_df_glacier["T2"] = input_df_glacier["T2"] + elev_diff_glacier * float(parameter.lr_temp)
        input_df_glacier["RRR"] = np.where(input_df_glacier["RRR"] > parameter.pfilter,         # Apply precipitation lapse rate only, when there is precipitation!
                                           input_df_glacier["RRR"] + elev_diff_glacier * float(parameter.lr_prec), 0)
        input_df_glacier["RRR"] = np.where(input_df_glacier["RRR"] < 0, 0, input_df_glacier["RRR"])
    else:
        input_df_glacier = df_preproc.copy()
    if parameter.ele_non_glac is not None:
        elev_diff_catchment = parameter.ele_non_glac - parameter.ele_dat
        input_df_catchment = df_preproc.copy()
        input_df_catchment["T2"] = input_df_catchment["T2"] + elev_diff_catchment * float(parameter.lr_temp)
        input_df_catchment["RRR"] = np.where(input_df_catchment["RRR"] > parameter.pfilter,     # Apply precipitation lapse rate only, when there is precipitation!
                                             input_df_catchment["RRR"] + elev_diff_catchment * float(parameter.lr_prec), 0)
        input_df_catchment["RRR"] = np.where(input_df_catchment["RRR"] < 0, 0, input_df_catchment["RRR"])

    else:
        input_df_catchment = df_preproc.copy()

    # precipitation phase separation:
    input_df_glacier['rain'], input_df_glacier['snow'] = phase_separation(input_df_glacier, parameter)
    input_df_catchment['rain'], input_df_catchment['snow'] = phase_separation(input_df_catchment, parameter)

    # snow correction factor
    input_df_glacier['snow'], input_df_catchment['snow'] = [parameter.SFCF * i for i in [input_df_glacier['snow'], input_df_catchment['snow']]]

    # add corrected snow fall to total precipitation
    input_df_glacier['RRR'], input_df_catchment['RRR'] = [i['snow'] + i['rain'] for i in [input_df_glacier, input_df_catchment]]

    return input_df_glacier, input_df_catchment


def calculate_PDD(ds, prints=True):
    """Calculation of positive degree days in the provided timeseries."""

    if prints:
        print('*-------------------*')
        print("Calculating positive degree days")

    # masking the dataset to glacier area
    if isinstance(ds, xr.Dataset):
        mask = ds.MASK.values
        temp = xr.where(mask == 1, ds["T2"], np.nan)
        temp = temp.mean(dim=["lat", "lon"])
        temp_mean = temp.resample(time="D").mean(dim="time")
        prec = xr.where(mask == 1, ds["RRR"], np.nan)
        prec = prec.mean(dim=["lat", "lon"])
        prec = prec.resample(time="D").sum(dim="time")
        rain = xr.where(mask == 1, ds["rain"], np.nan)
        rain = rain.mean(dim=["lat", "lon"])
        rain = rain.resample(time="D").sum(dim="time")
        snow = xr.where(mask == 1, ds["snow"], np.nan)
        snow = snow.mean(dim=["lat", "lon"])
        snow = snow.resample(time="D").sum(dim="time")
    else:
        temp = ds["T2"]
        temp_mean = temp.resample("D").mean()
        prec = ds["RRR"].resample("D").sum()
        rain = ds["rain"].resample("D").sum()
        snow = ds["snow"].resample("D").sum()


    pdd_ds = xr.merge([xr.DataArray(temp_mean, name="temp_mean"),
                       xr.DataArray(prec),
                       xr.DataArray(rain),
                       xr.DataArray(snow)])

    # calculate the positive degree days
    pdd_ds["pdd"] = xr.where(pdd_ds["temp_mean"] > 0, pdd_ds["temp_mean"], 0)

    return pdd_ds


def melt_rates(snow, pdd, parameter):
    """ pypdd.py line 331
        Compute melt rates from snow precipitation and pdd sum.
        Snow melt is computed from the number of positive degree days (*pdd*)
        and the `pdd_factor_snow` model attribute. If all snow is melted and
        some energy (PDD) remains, ice melt is computed using `pdd_factor_ice`.
        *snow*: array_like
            Snow precipitation rate.
        *pdd*: array_like
            Number of positive degree days."""

    # compute a potential snow melt
    pot_snow_melt = parameter.CFMAX_snow * pdd
    # effective snow melt can't exceed amount of snow
    snow_melt = np.minimum(snow, pot_snow_melt)
    # ice melt is proportional to excess snow melt
    ice_melt = (pot_snow_melt - snow_melt) * parameter.CFMAX_ice / parameter.CFMAX_snow
    # return melt rates
    return (snow_melt, ice_melt)


def calculate_glaciermelt(ds, parameter, prints=True):
    """Degree Day Model to calculate the accumulation, snow and ice melt and runoff rate from the glaciers.
    Roughly based on PYPDD (github.com/juseg/pypdd)
    - # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>)"""

    if prints:
        print("Calculating glacial melt")

    # initialize arrays
    temp = ds["temp_mean"].values
    prec = ds["RRR"].values
    snow = ds["snow"].values
    rain = ds["rain"].values
    pdd = ds["pdd"].values
    snow_depth = []
    snow_melt = []
    ice_melt = []
    actual_runoff = []
    glacier_reservoir = []

    accu_rate = snow

    # compute snow depth and melt rates
    for i in range(len(temp)):
        if i > 0:
            snow_depth.append(snow_depth[i - 1])
            snow_depth[i] += accu_rate[i]
        else:
            snow_depth.append(accu_rate[i])
        snow_melt_tmp, ice_melt_tmp = melt_rates(snow_depth[i], pdd[i], parameter)
        snow_melt.append(snow_melt_tmp)
        ice_melt.append(ice_melt_tmp)
        snow_depth[i] -= snow_melt_tmp

    # convert from list to array for arithmetic calculations
    snow_depth = np.array(snow_depth)   # not actual depth but mm w.e.!
    ice_melt = np.array(ice_melt)
    snow_melt = np.array(snow_melt)

    # calculate refreezing, runoff and surface mass balance
    total_melt = snow_melt + ice_melt
    refr_ice = parameter.CFR_ice * ice_melt
    refr_snow = parameter.RFS * snow_melt
    runoff_rate = total_melt - refr_snow - refr_ice
    inst_smb = accu_rate - runoff_rate
    runoff_rate_rain = runoff_rate + rain

    # Storage-release scheme for glacier outflow (Stahl et.al. 2008, Toum et. al. 2021)
    KG_min = 0.1  # minimum outflow coefficient (conditions with deep snow and poorly developed glacial drainage systems) [time^−1]
    d_KG = 0.9  # KG_min + d_KG = maximum outflow coefficient (representing late-summer conditions with bare ice and a well developed glacial drainage system) [time^−1]
    KG = np.minimum(KG_min + d_KG * np.exp(snow_depth / -(0.1 * 1000000**parameter.AG)), 1)
    for i in np.arange(len(temp)):
        if i == 0:
            SG = runoff_rate_rain[i]  # liquid water stored in the reservoir
        else:
            SG = np.maximum((runoff_rate_rain[i] - actual_runoff[i - 1]) + SG, 0)
        actual_runoff.append(KG[i] * SG)
        glacier_reservoir.append(SG)

    # final glacier module output (everything but temperature and pdd in mm w.e.)
    glacier_melt = xr.merge(
        [xr.DataArray(inst_smb, name="DDM_smb"),
         xr.DataArray(pdd, name="pdd"),
         xr.DataArray(temp, name="DDM_temp"),
         xr.DataArray(prec, name="DDM_prec"),
         xr.DataArray(rain, name="DDM_rain"),
         xr.DataArray(snow, name="DDM_snow"),
         xr.DataArray(accu_rate, name="DDM_accumulation_rate"),
         xr.DataArray(ice_melt, name="DDM_ice_melt"),
         xr.DataArray(snow_melt, name="DDM_snow_melt"),
         xr.DataArray(total_melt, name="DDM_total_melt"),
         xr.DataArray(refr_ice, name="DDM_refreezing_ice"),
         xr.DataArray(refr_snow, name="DDM_refreezing_snow"),
         xr.DataArray(glacier_reservoir, name="DDM_glacier_reservoir"),
         xr.DataArray(actual_runoff, name='Q_DDM')
         ])

    DDM_results = glacier_melt.to_dataframe()

    # merged data array comes without index -> set DateTime index from input
    idx = ds.coords.to_index()
    DDM_results = DDM_results.set_index(pd.DatetimeIndex(idx))

    if prints:
        print("Finished Degree-Day Melt Routine")

    return DDM_results


def create_lookup_table(glacier_profile, parameter):
    """ Part 1 of the glacier scaling routine based on the deltaH approach outlined in Seibert et al. (2018) and
    Huss and al.(2010). Creates a look-up table of glacier area and water equivalent from the initial state (100%)
    to an ice-free catchment (0%) in steps of 1%."""

    initial_area = glacier_profile["Area"]  # per elevation band
    hi_initial = glacier_profile["WE"]  # initial water equivalent of each elevation band
    hi_k = glacier_profile["WE"]  # hi_k is the updated water equivalent for each elevation zone, starts with initial values
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
    if parameter.area_glac < 5:
        a = -0.3
        b = 0.6
        c = 0.09
        y = 2
    elif parameter.area_glac < 20:
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

    ai_scaled = ai.copy()  # initial values set as ai_scaled 

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


def glacier_area_change(output_DDM, lookup_table, glacier_profile, parameter):
    """ Part 2 of the glacier scaling routine based on the deltaH approach outlined in Seibert et al. (2018) and
    Huss and al.(2010). Calculates the new glacier area for each hydrological year."""

    # select output columns to update
    up_cols = output_DDM.columns.drop(['DDM_smb', 'DDM_temp', 'pdd'])

    # creating columns for updated DDM output
    for col in up_cols:
        output_DDM[col + '_updated_scaled'] = copy.deepcopy(output_DDM[col])

    # determining the hydrological year
    output_DDM["water_year"] = np.where((output_DDM.index.month) >= parameter.hydro_year, output_DDM.index.year + 1,
                                        output_DDM.index.year)

    # initial glacier mass from the glacier profile in mm w.e. (relative to the whole catchment)
    m = sum((glacier_profile["Area"]) * glacier_profile["WE"])

    # initial area
    initial_area = glacier_profile.groupby("EleZone")["Area"].sum()

    # dataframe with the smb change per hydrological year in mm w.e.
    glacier_change = pd.DataFrame({"smb": output_DDM.groupby("water_year")["DDM_smb"].sum()}).reset_index()

    glacier_change_area = pd.DataFrame({"time": "initial", "glacier_area": [parameter.area_glac]})

    # setting initial values for the loop
    new_area = parameter.area_glac
    smb_cum = 0
    i = 1
    for i in range(len(glacier_change)):
        year = glacier_change["water_year"][i]
        smb = glacier_change["smb"][i]
        # scale the smb to the (updated) glacierized fraction of the catchment
        smb = smb * (new_area / parameter.area_cat)       # SMB is area (re-)scaled because m is area scaled as well
        # add the smb from the previous year(s) to the new year
        smb_cum = smb_cum + smb
        # calculate the percentage of melt in comparison to the initial mass
        smb_percentage = round((-smb_cum / m) * 100)
        if (smb_percentage <= 99) & (smb_percentage >= 0):
            # select the correct row from the lookup table depending on the smb
            area_melt = lookup_table.iloc[smb_percentage]
            # derive the new glacier area by multiplying the initial area with the area changes
            new_area = np.nansum((area_melt.values * initial_area.values))*parameter.area_cat
        else:
            new_area = 0
        # scale the output with the new glacierized area
        glacier_change_area = glacier_change_area.append({
            'time': year, "glacier_area": new_area, "smb_scaled_cum": smb_cum
        }, ignore_index=True)
        for col in up_cols:
            output_DDM[col + "_updated_scaled"] = np.where(output_DDM["water_year"] == year, output_DDM[col] * (new_area / parameter.area_cat), output_DDM[col + "_updated_scaled"])

    return output_DDM, glacier_change_area


def updated_glacier_melt(data, lookup_table, glacier_profile, parameter):
    """Function to account for the elevation change due to retreating or advancing glaciers. Runs scaling and melt
    routines on single hydrological years continuously updating the glacierized catchment fraction and mean glacier
    elevation altered by the deltaH routine. Slightly increases processing time due to the use of standard loops."""

    # determine hydrological years
    data["water_year"] = np.where((data.index.month) >= parameter.hydro_year, data.index.year + 1, data.index.year)

    # initial glacier mass from the glacier profile in mm w.e. (relative to the whole catchment)
    m = sum((glacier_profile["Area"]) * glacier_profile["WE"])

    # initial area
    initial_area = glacier_profile.groupby("EleZone")["Area"].sum()

    # re-calculate the mean glacier elevation based on the glacier profile in rough elevation zones for consistency (method outlined in following loop)
    print("Recalculating initial elevations based on glacier profile")
    init_dist = initial_area.values / initial_area.values.sum()     # fractions of glacierized area elev zones
    init_elev = init_dist * lookup_table.columns.values             # multiply fractions with average zone elevations
    init_elev = int(init_elev.sum())
    print(">> Prior glacier elevation: " + str(parameter.ele_glac) + 'm a.s.l.')
    print(">> Recalculated glacier elevation: " + str(init_elev) + 'm a.s.l.')

    # re-calculate the mean non-glacierized elevation accordingly
    ele_non_glac = (parameter.ele_cat - parameter.area_glac
                                      / parameter.area_cat * init_elev) \
                                     * parameter.area_cat / (parameter.area_cat
                                                                     - parameter.area_glac)

    print(">> Prior non-glacierized elevation: " + str(round(parameter.ele_non_glac)) + 'm a.s.l.')
    print(">> Recalculated non-glacierized elevation: " + str(round(ele_non_glac)) + 'm a.s.l.')

    # create initial df of glacier change
    glacier_change = pd.DataFrame({"time": "initial", "glacier_area": [parameter.area_glac],
                                        "glacier_elev": init_elev})

    # Setup initial variables for main loop
    new_area = parameter.area_glac
    smb_cum = 0
    output_DDM = pd.DataFrame()
    parameter_updated = copy.deepcopy(parameter)
    parameter_updated.ele_glac = init_elev
    parameter_updated.ele_non_glac = ele_non_glac

    # create initial non-updated dataframes
    input_df_glacier, input_df_catchment = input_scaling(data, parameter_updated)

    # Slice input data to simulation period (with full hydrological years if the setup period allows it)
    if datetime.fromisoformat(parameter.sim_start).month < parameter.hydro_year:
        startyear = data[parameter.sim_start:parameter.sim_end].water_year[0] - 1
    else:
        startyear = data[parameter.sim_start:parameter.sim_end].water_year[0]

    startdate = str(startyear) + '-' + str(parameter.hydro_year) + '-' + '01'

    if datetime.fromisoformat(startdate) < datetime.fromisoformat(parameter.set_up_start):
        # Provided setup period does not cover the full hydrological year sim_start falls in
        data_update = data[parameter.sim_start:parameter.sim_end]
        input_df_glacier = input_df_glacier[parameter.sim_start:parameter.sim_end]
        input_df_catchment_spinup = input_df_catchment[parameter.set_up_start:parameter.sim_start]
        input_df_catchment = input_df_catchment[parameter.sim_start:parameter.sim_end]


        print("WARNING! The provided setup period does not cover the full hydrological year the simulation period"
              "starts in. The initial surface mass balance (SMB) of the first hydrological year in the glacier "
              "rescaling routine therefore possibly misses a significant part of the accumulation period (e.g. Oct-Dec).")
    else:
        data_update = data[startdate:parameter.sim_end]
        input_df_glacier = input_df_glacier[startdate:parameter.sim_end]
        input_df_catchment_spinup = input_df_catchment[parameter.set_up_start:startdate]
        input_df_catchment = input_df_catchment[startdate:parameter.sim_end]

    # MAIN LOOP
    # Loop through simulation period annually updating catchment fractions and scaling elevations
    if parameter.ele_dat is not None:

        print('Calculating glacier evolution')
        for i in range(len(data_update.water_year.unique())):
            year = data_update.water_year.unique()[i]
            mask = data_update.water_year == year

            # Use updated glacier area of the previous year
            parameter_updated.area_glac = new_area
            # Use updated glacier elevation of the previous year
            if i is not 0:
                parameter_updated.ele_glac = new_distribution

            # Calculate the updated mean elevation of the non-glacierized catchment area
            parameter_updated.ele_non_glac = (parameter_updated.ele_cat - parameter_updated.area_glac
                                             / parameter_updated.area_cat * parameter_updated.ele_glac) \
                                             * parameter_updated.area_cat / (parameter_updated.area_cat
                                                                             - parameter_updated.area_glac)


            # Scale glacier and hbv routine inputs in selected year with updated parameters
            input_df_glacier[mask], input_df_catchment[mask] = input_scaling(data_update[mask], parameter_updated)

            # Calculate positive degree days and glacier ablation/accumulation
            degreedays_ds = calculate_PDD(input_df_glacier[mask], prints=False)
            output_DDM_year = calculate_glaciermelt(degreedays_ds, parameter_updated, prints=False)
            output_DDM_year['water_year'] = data_update.water_year[mask]

            # select output columns to update
            up_cols = output_DDM_year.columns.drop(['DDM_smb', 'DDM_temp', 'pdd', 'water_year'])
            # create columns for updated DDM output
            for col in up_cols:
                output_DDM_year[col + '_updated_scaled'] = copy.deepcopy(output_DDM_year[col])

            # Rescale glacier geometry and update glacier parameters in all but the last (incomplete) water year
            if i < len(data_update.water_year.unique()) - 1:

                smb_unscaled = output_DDM_year["DDM_smb"].sum()
                if i is 0 and smb_unscaled > 0:
                    print("ERROR: The cumulative surface mass balance in the first year of the simulation period is "
                          "positive. You may want to shift the starting year.")
                # scale the smb to the (updated) glacierized fraction of the catchment
                smb = smb_unscaled * (new_area / parameter.area_cat)  # SMB is area (re-)scaled because m is area scaled as well
                # add the smb from the previous year(s) to the new year
                smb_cum = smb_cum + smb
                if smb_cum > 0:
                    print("ERROR: The cumulative surface mass balance in the simulation period is positive. "
                          "The glacier rescaling routine cannot model glacier extent exceeding the initial status of "
                          "the provided glacier profile. In order to exclude this run from parameter optimization "
                          "routines, a flag is passed and simulated runoff is set to 0.01.")
                    smb_cum = m
                    new_distribution = parameter.ele_glac
                    smb_flag = True
                else:
                    smb_flag = False
                # calculate the percentage of melt in comparison to the initial mass
                smb_percentage = round((-smb_cum / m) * 100)
                if (smb_percentage < 99) & (smb_percentage >= 0):
                    # select the correct row from the lookup table depending on the smb
                    area_melt = lookup_table.iloc[smb_percentage]
                    # derive the new glacier area by multiplying the initial area with the area changes
                    new_area = np.nansum((area_melt.values * initial_area.values)) * parameter.area_cat
                    # derive new spatial distribution of glacierized area (relative fraction in every elevation zone)
                    new_distribution = ((area_melt.values * initial_area.values) * parameter.area_cat) / new_area
                    # multiply relative portions with mean zone elevations to get rough estimate for new mean elevation
                    new_distribution = new_distribution * lookup_table.columns.values  # column headers contain elevations
                    new_distribution = int(new_distribution.sum())
                else:
                    new_area = 0

                # Create glacier change dataframe for subsequent functions (skip last incomplete year)
                glacier_change = pd.concat([glacier_change, pd.DataFrame({
                    'time': year, "glacier_area": new_area, "glacier_elev": new_distribution, 'smb_water_year': smb_unscaled,
                    "smb_scaled_cum": smb_cum}, index=[i])], ignore_index=True)

            # Scale DDM output to new glacierized fraction
            for col in up_cols:
                output_DDM_year[col + "_updated_scaled"] = np.where(output_DDM_year["water_year"] == year,
                                                               output_DDM_year[col] * (new_area / parameter.area_cat),
                                                               output_DDM_year[col + "_updated_scaled"])
            # Append year to full dataset
            output_DDM = pd.concat([output_DDM, output_DDM_year])

            if smb_flag:
                output_DDM['smb_flag'] = 1

        output_DDM = output_DDM[parameter.sim_start:parameter.sim_end]
        # Add spinup original spin-up period back to HBV input
        input_df_catchment = pd.concat([input_df_catchment_spinup, input_df_catchment])

        return output_DDM, glacier_change, input_df_catchment

    else:
        print("ERROR: You need to provide ele_dat in order to apply the glacier-rescaling routine.")
        return


def hbv_simulation(input_df_catchment, parameter, glacier_area=None):
        """Compute the runoff from the catchment with the HBV model
            Python Code based on the LHMP (github.com/hydrogo/LHMP -
            Ayzel Georgy. (2016). LHMP: lumped hydrological modelling playground. Zenodo. doi: 10.5281/zenodo.59501)
            For the HBV model, evapotranspiration values are needed. If none provided these are calculated as suggested by Oudin et al. (2005)
            in mm/day."""
        print('*-------------------*')
        print("Running HBV routine")
        # 1. new temporary dataframe from input with daily values
        if "PE" in input_df_catchment.columns:
            input_df_hbv = input_df_catchment.resample("D").agg({"T2": 'mean', "RRR": 'sum', "rain": "sum",
                                                                 "snow": "sum", "PE": "sum"})
        else:
            input_df_hbv = input_df_catchment.resample("D").agg({"T2": 'mean', "RRR": 'sum', "rain": "sum",
                                                                 "snow": "sum"})

        Temp = input_df_hbv['T2']
        Prec = input_df_hbv['RRR']
        rain = input_df_hbv['rain']
        snow = input_df_hbv['snow']

        # Calculation of PE with Oudin et al. 2005
        latent_heat_flux = 2.45
        water_density = 1000
        if "PE" in input_df_catchment.columns:
            Evap = input_df_hbv["PE"]
        else:
            doy = np.array(input_df_hbv.index.strftime('%j')).astype(int)
            lat = np.deg2rad(parameter.lat)

            # Part 2. Extraterrestrial radiation calculation
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
            Re = Rsc * 86400 / np.pi * dr * (ws * np.sin(lat) * np.sin(dt)
                                             + np.sin(ws) * np.cos(lat) * np.cos(dt))
            # convert from J m-2 day-1 to MJ m-2 day-1
            Re = Re / 10 ** 6

            Evap = np.where(Temp + 5 > 0, (Re / (water_density * latent_heat_flux)) * ((Temp + 5) / 100) * 1000, 0)

            Evap = pd.Series(Evap, index=input_df_hbv.index)
            input_df_hbv["PE"] = Evap

        # 2. Set-up period:
        # 2.1 meteorological forcing preprocessing
        Temp_cal = Temp[parameter.set_up_start:parameter.set_up_end]
        Prec_cal = Prec[parameter.set_up_start:parameter.set_up_end]
        SNOW_cal = snow[parameter.set_up_start:parameter.set_up_end]
        RAIN_cal = rain[parameter.set_up_start:parameter.set_up_end]
        Evap_cal = Evap[parameter.set_up_start:parameter.set_up_end]

        # get the new glacier area for each year      --> I think this section is redundant. glacier_area does not cover the set_up period!
        if glacier_area is not None:
            glacier_area = glacier_area.iloc[1:, :]
            glacier_area["time"] = glacier_area["time"].astype(str).astype(float).astype(int)
            SNOW2 = pd.DataFrame(SNOW_cal)
            SNOW2["area"] = 0
            for year in range(len(glacier_area)):
                SNOW2["area"] = np.where(SNOW2.index.year == glacier_area["time"].iloc[year],
                                         glacier_area["glacier_area"].iloc[year],
                                         SNOW2["area"])

            SNOW2["snow"] = SNOW2["snow"] * (1 - (SNOW2["area"] / parameter.area_cat))
            SNOW_cal = SNOW2['snow'].squeeze()
            RAIN_cal = RAIN_cal * (1 - (SNOW2["area"] / parameter.area_cat))
            Prec_cal = Prec_cal * (1 - (SNOW2["area"] / parameter.area_cat))
        else:
            SNOW_cal = SNOW_cal * (1 - (parameter.area_glac / parameter.area_cat))
            RAIN_cal = RAIN_cal * (1 - (parameter.area_glac / parameter.area_cat))
            Prec_cal = Prec_cal * (1 - (parameter.area_glac / parameter.area_cat))

        # evaporation correction
        # a. calculate long-term averages of daily temperature
        Temp_mean_cal = np.array([Temp_cal.loc[Temp_cal.index.dayofyear == x].mean() \
                                  for x in range(1, 367)])
        # b. correction of Evaporation daily values
        Evap_cal = Evap_cal.index.map(
            lambda x: (1 + parameter.CET * (Temp_cal[x] - Temp_mean_cal[x.dayofyear - 1])) * Evap_cal[x])
        # c. control Evaporation
        Evap_cal = np.where(Evap_cal > 0, Evap_cal, 0)

        # 2.2 Initial parameter calibration
        # snowpack box
        SNOWPACK_cal = np.zeros(len(Prec_cal))
        SNOWPACK_cal[0] = 0.0001
        # meltwater box
        SNOWMELT_cal = np.zeros(len(Prec_cal))
        SNOWMELT_cal[0] = 0.0001
        # soil moisture box
        SM_cal = np.zeros(len(Prec_cal))
        SM_cal[0] = 0.0001
        # actual evaporation
        ETact_cal = np.zeros(len(Prec_cal))
        ETact_cal[0] = 0.0001

        # 2.3 Running model for set-up period
        for t in range(1, len(Prec_cal)):

            # 2.3.1 Snow routine
            # how snowpack forms
            SNOWPACK_cal[t] = SNOWPACK_cal[t - 1] + SNOW_cal[t]
            # how snowpack melts
            # day-degree simple melting
            melt = parameter.CFMAX_snow * (Temp_cal[t] - parameter.TT_snow)
            # control melting
            if melt < 0: melt = 0
            melt = min(melt, SNOWPACK_cal[t])
            # how meltwater box forms
            SNOWMELT_cal[t] = SNOWMELT_cal[t - 1] + melt
            # snowpack after melting
            SNOWPACK_cal[t] = SNOWPACK_cal[t] - melt
            # refreezing accounting
            refreezing = parameter.RFS * parameter.CFMAX_snow * (parameter.TT_snow - Temp_cal[t])
            # control refreezing
            if refreezing < 0: refreezing = 0
            refreezing = min(refreezing, SNOWMELT_cal[t])
            # snowpack after refreezing
            SNOWPACK_cal[t] = SNOWPACK_cal[t] + refreezing
            # meltwater after refreezing
            SNOWMELT_cal[t] = SNOWMELT_cal[t] - refreezing
            # recharge to soil
            tosoil = SNOWMELT_cal[t] - (parameter.CWH * SNOWPACK_cal[t]);
            # control recharge to soil
            if tosoil < 0: tosoil = 0
            # meltwater after recharge to soil
            SNOWMELT_cal[t] = SNOWMELT_cal[t] - tosoil

            # 2.3.1 Soil and evaporation routine
            # soil wetness calculation
            soil_wetness = (SM_cal[t - 1] / parameter.FC) ** parameter.BETA
            # control soil wetness (should be in [0, 1])
            if soil_wetness < 0: soil_wetness = 0
            if soil_wetness > 1: soil_wetness = 1
            # soil recharge
            recharge = (RAIN_cal[t] + tosoil) * soil_wetness
            # soil moisture update
            SM_cal[t] = SM_cal[t - 1] + RAIN_cal[t] + tosoil - recharge
            # excess of water calculation
            excess = SM_cal[t] - parameter.FC
            # control excess
            if excess < 0: excess = 0
            # soil moisture update
            SM_cal[t] = SM_cal[t] - excess

            # evaporation accounting
            evapfactor = SM_cal[t] / (parameter.LP * parameter.FC)
            # control evapfactor in range [0, 1]
            if evapfactor < 0: evapfactor = 0
            if evapfactor > 1: evapfactor = 1
            # calculate actual evaporation
            ETact_cal[t] = Evap_cal[t] * evapfactor
            # control actual evaporation
            ETact_cal[t] = min(SM_cal[t], ETact_cal[t])

            # last soil moisture updating
            SM_cal[t] = SM_cal[t] - ETact_cal[t]
        print("Finished spin up for initial HBV parameters")

        # 3. meteorological forcing preprocessing for simulation
        Temp = Temp[parameter.sim_start:parameter.sim_end]
        Prec = Prec[parameter.sim_start:parameter.sim_end]
        SNOW = snow[parameter.sim_start:parameter.sim_end]
        RAIN = rain[parameter.sim_start:parameter.sim_end]
        Evap = Evap[parameter.sim_start:parameter.sim_end]

        # get the new glacier area for each year
        if glacier_area is not None:
            glacier_area = glacier_area.iloc[1:, :]
            glacier_area["time"] = glacier_area["time"].astype(str).astype(float).astype(int)
            SNOW2 = pd.DataFrame(SNOW)
            SNOW2["area"] = 0
            for year in range(len(glacier_area)):
                SNOW2["area"] = np.where(SNOW2.index.year == glacier_area["time"].iloc[year],
                                         glacier_area["glacier_area"].iloc[year],
                                         SNOW2["area"])
            RAIN = RAIN * (1 - (SNOW2["area"] / parameter.area_cat))            # Rain off glacier
            SNOW2["snow"] = SNOW2["snow"] * (1 - (SNOW2["area"] / parameter.area_cat))  # Snow off-glacier
            SNOW = SNOW2['snow'].squeeze()
            Prec = Prec * (1 - (SNOW2["area"] / parameter.area_cat))

        else:
            RAIN = RAIN * (1 - (parameter.area_glac / parameter.area_cat))    # Rain off glacier
            SNOW = SNOW * (1 - (parameter.area_glac / parameter.area_cat))    # Snow off-glacier
            Prec = Prec * (1 - (parameter.area_glac / parameter.area_cat))

        # a. calculate long-term averages of daily temperature
        Temp_mean = np.array([Temp.loc[Temp.index.dayofyear == x].mean() for x in range(1, 367)])
        # b. correction of Evaporation daily values
        Evap = Evap.index.map(lambda x: (1 + parameter.CET * (Temp[x] - Temp_mean[x.dayofyear - 1])) * Evap[x])
        # c. control Evaporation
        Evap = np.where(Evap > 0, Evap, 0)

        # 4. initialize boxes and initial conditions after calibration
        # snowpack box
        SNOWPACK = np.zeros(len(Prec))
        SNOWPACK[0] = SNOWPACK_cal[-1]
        # meltwater box
        SNOWMELT = np.zeros(len(Prec))
        SNOWMELT[0] = 0.0001
        # total melt off-glacier
        off_glac = np.zeros(len(Prec))
        off_glac[0] = 0.0001
        # soil moisture box
        SM = np.zeros(len(Prec))
        SM[0] = SM_cal[-1]
        # soil upper zone box
        SUZ = np.zeros(len(Prec))
        SUZ[0] = 0.0001
        # soil lower zone box
        SLZ = np.zeros(len(Prec))
        SLZ[0] = 0.0001
        # actual evaporation
        ETact = np.zeros(len(Prec))
        ETact[0] = 0.0001
        # simulated runoff box
        Qsim = np.zeros(len(Prec))
        Qsim[0] = 0.0001

        # 5. The main cycle of calculations
        for t in range(1, len(Qsim)):
            # 5.1 Snow routine
            # how snowpack forms
            SNOWPACK[t] = SNOWPACK[t - 1] + SNOW[t]
            # how snowpack melts
            # temperature index melting (PDD)
            melt = parameter.CFMAX_snow * (Temp[t] - parameter.TT_snow)
            # control melting
            if melt < 0: melt = 0
            melt = min(melt, SNOWPACK[t])
            # how meltwater box forms
            SNOWMELT[t] = SNOWMELT[t - 1] + melt
            # snowpack after melting
            SNOWPACK[t] = SNOWPACK[t] - melt
            # refreezing accounting
            refreezing = parameter.RFS * parameter.CFMAX_snow * (parameter.TT_snow - Temp[t])
            # control refreezing
            if refreezing < 0: refreezing = 0
            refreezing = min(refreezing, SNOWMELT[t])
            # snowpack after refreezing
            SNOWPACK[t] = SNOWPACK[t] + refreezing
            # meltwater after refreezing
            SNOWMELT[t] = SNOWMELT[t] - refreezing
            # Total melt off-glacier
            off_glac[t] = SNOWMELT[t]
            # recharge to soil
            tosoil = SNOWMELT[t] - (parameter.CWH * SNOWPACK[t])
            # control recharge to soil
            if tosoil < 0: tosoil = 0
            # meltwater after recharge to soil
            SNOWMELT[t] = SNOWMELT[t] - tosoil

            # 5.2 Soil and evaporation routine
            # soil wetness calculation
            soil_wetness = (SM[t - 1] / parameter.FC) ** parameter.BETA
            # control soil wetness (should be in [0, 1])
            if soil_wetness < 0: soil_wetness = 0
            if soil_wetness > 1: soil_wetness = 1
            # soil recharge
            recharge = (RAIN[t] + tosoil) * soil_wetness

            # soil moisture update
            SM[t] = SM[t - 1] + RAIN[t] + tosoil - recharge
            # excess of water calculation
            excess = SM[t] - parameter.FC
            # control excess
            if excess < 0: excess = 0
            # soil moisture update
            SM[t] = SM[t] - excess

            # evaporation accounting
            evapfactor = SM[t] / (parameter.LP * parameter.FC)
            # control evapfactor in range [0, 1]
            if evapfactor < 0: evapfactor = 0
            if evapfactor > 1: evapfactor = 1
            # calculate actual evaporation
            ETact[t] = Evap[t] * evapfactor
            # control actual evaporation
            ETact[t] = min(SM[t], ETact[t])

            # last soil moisture updating
            SM[t] = SM[t] - ETact[t]

            # 5.3 Groundwater routine
            # upper groundwater box
            SUZ[t] = SUZ[t - 1] + recharge + excess
            # percolation control
            perc = min(SUZ[t], parameter.PERC)
            # update upper groundwater box
            SUZ[t] = SUZ[t] - perc
            # runoff from the highest part of upper groundwater box (surface runoff)
            Q0 = parameter.K0 * max(SUZ[t] - parameter.UZL, 0)
            # update upper groundwater box
            SUZ[t] = SUZ[t] - Q0
            # runoff from the middle part of upper groundwater box
            Q1 = parameter.K1 * SUZ[t]
            # update upper groundwater box
            SUZ[t] = SUZ[t] - Q1
            # calculate lower groundwater box
            SLZ[t] = SLZ[t - 1] + perc
            # runoff from lower groundwater box
            Q2 = parameter.K2 * SLZ[t]
            # update lower groundwater box
            SLZ[t] = SLZ[t] - Q2

            # Total runoff calculation
            Qsim[t] = Q0 + Q1 + Q2

        # 6. Scale effect accounting
        # delay and smoothing simulated hydrograph
        # (Beck et al.,2016) used triangular transformation based on moving window
        # here are my method with simple forward filter based on Butterworth filter design
        # calculate Numerator (b) and denominator (a) polynomials of the IIR filter
        parMAXBAS = int(parameter.MAXBAS)
        b, a = ss.butter(parMAXBAS, 1 / parMAXBAS)
        # implement forward filter
        Qsim_smoothed = ss.lfilter(b, a, Qsim)
        # control smoothed runoff
        Qsim_smoothed = np.where(Qsim_smoothed > 0, Qsim_smoothed, 0)

        Qsim = Qsim_smoothed
        hbv_results = pd.DataFrame(
            {"HBV_temp": Temp, "HBV_prec": Prec, "HBV_rain": RAIN, "HBV_snow": SNOW, "HBV_pe": Evap,
             "HBV_snowpack": SNOWPACK, "HBV_soil_moisture": SM, "HBV_AET": ETact, "HBV_refreezing": refreezing,
             "HBV_upper_gw": SUZ, "HBV_lower_gw": SLZ, "HBV_melt_off_glacier": off_glac, "Q_HBV": Qsim},
            index=input_df_hbv[parameter.sim_start: parameter.sim_end].index)
        print("Finished HBV routine")
        return hbv_results


def create_statistics(output_MATILDA):
    stats = output_MATILDA.describe()
    sum = pd.DataFrame(output_MATILDA.sum())
    sum.columns = ["sum"]
    sum = sum.transpose()
    stats = stats.append(sum)
    stats = stats.round(3)
    return stats


def matilda_submodules(df_preproc, parameter, obs=None, glacier_profile=None, elev_rescaling=False):
    """The main MATILDA simulation. It applies a linear scaling of the data (if elevations
    are provided) and executes the DDM and HBV modules subsequently."""

    # Filter warnings:
    if not parameter.warn:
        warnings.filterwarnings(action='ignore')

    print('---')
    print('Initiating MATILDA simulation')

    # Rescale glacier elevation or not?
    if elev_rescaling:
        # Execute glacier-rescaling module
        if parameter.area_glac > 0:
            if glacier_profile is not None:
                lookup_table = create_lookup_table(glacier_profile, parameter)
                output_DDM, glacier_change, input_df_catchment = updated_glacier_melt(df_preproc, lookup_table,
                                                                                                  glacier_profile,
                                                                                                  parameter)
            else:
                print("ERROR: No glacier profile passed for glacier elevation rescaling! Provide a glacier profile or"
                      " set elev_rescaling=False")
                return
        else:
            lookup_table = str("No lookup table generated")
            glacier_change = str("No glacier changes calculated")

    else:
        print("WARNING: Glacier elevation scaling is turned off. The average glacier elevation is treated as constant. "
              "This might cause a significant bias in glacier melt on larger time scales! Set elev_rescaling=True "
              "to annually rescale glacier elevations.")
        # Scale input data to fit catchments elevations
        if parameter.ele_dat is not None:
            input_df_glacier, input_df_catchment = input_scaling(df_preproc, parameter)
        else:
            input_df_glacier = df_preproc.copy()
            input_df_catchment = df_preproc.copy()

        input_df_glacier = input_df_glacier[parameter.sim_start:parameter.sim_end]

        # Execute DDM module
        if parameter.area_glac > 0:
            degreedays_ds = calculate_PDD(input_df_glacier)
            output_DDM = calculate_glaciermelt(degreedays_ds, parameter)

        # Execute glacier re-scaling module
        if parameter.area_glac > 0:
            if glacier_profile is not None:
                lookup_table = create_lookup_table(glacier_profile, parameter)
                output_DDM, glacier_change = glacier_area_change(output_DDM, lookup_table, glacier_profile, parameter)
            else:
                # scaling DDM output to fraction of catchment area
                for col in output_DDM.columns.drop(['DDM_smb','pdd']):
                     output_DDM[col + "_scaled"] = output_DDM[col] * (parameter.area_glac / parameter.area_cat)

                lookup_table = str("No lookup table generated")
                glacier_change = str("No glacier changes calculated")
        else:
            lookup_table = str("No lookup table generated")
            glacier_change = str("No glacier changes calculated")

    # Execute HBV module:
    if glacier_profile is not None:
        output_HBV = hbv_simulation(input_df_catchment, parameter, glacier_area=glacier_change)
    else:
        output_HBV = hbv_simulation(input_df_catchment, parameter)
    output_HBV = output_HBV[parameter.sim_start:parameter.sim_end]

    # Output postprocessing
    if parameter.area_glac > 0:
        output_MATILDA = pd.concat([output_HBV, output_DDM], axis=1)
    else:
        output_MATILDA = output_HBV.copy()

    if obs is not None:
        output_MATILDA = pd.concat([output_MATILDA, obs], axis=1)

    if parameter.area_glac > 0:
        if glacier_profile is not None:
            output_MATILDA["Q_Total"] = output_MATILDA["Q_HBV"] + output_MATILDA["Q_DDM_updated_scaled"]
            output_MATILDA["Prec_total"] = output_MATILDA["DDM_rain_updated_scaled"] +\
                                           output_MATILDA["DDM_snow_updated_scaled"] +\
                                           output_MATILDA["HBV_rain"] +\
                                           output_MATILDA["HBV_snow"]
            output_MATILDA["Melt_total"] = output_MATILDA["DDM_total_melt_updated_scaled"] +\
                                           output_MATILDA["HBV_melt_off_glacier"]
        else:
            output_MATILDA["Q_Total"] = output_MATILDA["Q_HBV"] + output_MATILDA["Q_DDM_scaled"]
            output_MATILDA["Prec_total"] = output_MATILDA["DDM_rain_scaled"] + \
                                           output_MATILDA["DDM_snow_scaled"] + \
                                           output_MATILDA["HBV_rain"] + \
                                           output_MATILDA["HBV_snow"]
            output_MATILDA["Melt_total"] = output_MATILDA["DDM_total_melt_scaled"] +\
                                           output_MATILDA["HBV_melt_off_glacier"]
    else:
        output_MATILDA["Q_Total"] = output_MATILDA["Q_HBV"]

    output_MATILDA = output_MATILDA[parameter.sim_start:parameter.sim_end]

    if "smb_flag" in output_MATILDA.columns:
        output_MATILDA['Q_Total'] = 0.01

    # Add compact output
    if parameter.area_glac > 0:
        if glacier_profile is not None:
            output_MATILDA_compact = pd.DataFrame(
                {'avg_temp_catchment': output_MATILDA['HBV_temp'],
                'avg_temp_glaciers': output_MATILDA['DDM_temp'],
                'evap_off_glaciers': output_MATILDA['HBV_pe'],
                'prec_off_glaciers': output_MATILDA['HBV_prec'],
                'prec_on_glaciers': output_MATILDA['DDM_prec_updated_scaled'],
                'rain_off_glaciers': output_MATILDA['HBV_rain'],
                'snow_off_glaciers': output_MATILDA['HBV_snow'],
                'rain_on_glaciers': output_MATILDA['DDM_rain_updated_scaled'],
                'snow_on_glaciers': output_MATILDA['DDM_snow_updated_scaled'],
                'snowpack_off_glaciers': output_MATILDA['HBV_snowpack'],
                'soil_moisture': output_MATILDA['HBV_soil_moisture'],
                'upper_groundwater': output_MATILDA['HBV_upper_gw'],
                'lower_groundwater': output_MATILDA['HBV_lower_gw'],
                'melt_off_glaciers': output_MATILDA['HBV_melt_off_glacier'],
                'melt_on_glaciers': output_MATILDA['DDM_total_melt_updated_scaled'],
                'ice_melt_on_glaciers': output_MATILDA['DDM_ice_melt_updated_scaled'],
                'snow_melt_on_glaciers': output_MATILDA['DDM_snow_melt_updated_scaled'],
                'refreezing_ice': output_MATILDA['DDM_refreezing_ice_updated_scaled'],
                'refreezing_snow': output_MATILDA['DDM_refreezing_snow_updated_scaled'],
                'total_refreezing': output_MATILDA['DDM_refreezing_ice_updated_scaled'] + output_MATILDA['DDM_refreezing_snow_updated_scaled'] + output_MATILDA['HBV_refreezing'],
                'SMB': output_MATILDA['DDM_smb'],
                'actual_evaporation': output_MATILDA['HBV_AET'],
                'total_precipitation': output_MATILDA['Prec_total'],
                'total_melt': output_MATILDA['Melt_total'],
                'runoff_without_glaciers': output_MATILDA['Q_HBV'],
                'runoff_from_glaciers': output_MATILDA['Q_DDM_updated_scaled'],
                'total_runoff': output_MATILDA['Q_Total'],
                'observed_runoff': output_MATILDA['Qobs']}, index=output_MATILDA.index)
        else:
            output_MATILDA_compact = pd.DataFrame(
                {'avg_temp_catchment': output_MATILDA['HBV_temp'],
                'avg_temp_glaciers': output_MATILDA['DDM_temp'],
                'evap_off_glaciers': output_MATILDA['HBV_pe'],
                'prec_off_glaciers': output_MATILDA['HBV_prec'],
                'prec_on_glaciers': output_MATILDA['DDM_prec_scaled'],
                'rain_off_glaciers': output_MATILDA['HBV_rain'],
                'snow_off_glaciers': output_MATILDA['HBV_snow'],
                'rain_on_glaciers': output_MATILDA['DDM_rain_scaled'],
                'snow_on_glaciers': output_MATILDA['DDM_snow_scaled'],
                'snowpack_off_glaciers': output_MATILDA['HBV_snowpack'],
                'soil_moisture': output_MATILDA['HBV_soil_moisture'],
                'upper_groundwater': output_MATILDA['HBV_upper_gw'],
                'lower_groundwater': output_MATILDA['HBV_lower_gw'],
                'melt_off_glaciers': output_MATILDA['HBV_melt_off_glacier'],
                'melt_on_glaciers': output_MATILDA['DDM_total_melt_scaled'],
                'ice_melt_on_glaciers': output_MATILDA['DDM_ice_melt_scaled'],
                'snow_melt_on_glaciers': output_MATILDA['DDM_snow_melt_scaled'],
                'refreezing_ice': output_MATILDA['DDM_refreezing_ice_scaled'],
                'refreezing_snow': output_MATILDA['DDM_refreezing_snow_scaled'],
                'total_refreezing': output_MATILDA['DDM_refreezing_ice_scaled'] + output_MATILDA['DDM_refreezing_snow_scaled'] + output_MATILDA['HBV_refreezing'],
                'SMB': output_MATILDA['DDM_smb'],
                'actual_evaporation': output_MATILDA['HBV_AET'],
                'total_precipitation': output_MATILDA['Prec_total'],
                'total_melt': output_MATILDA['Melt_total'],
                'runoff_without_glaciers': output_MATILDA['Q_HBV'],
                'runoff_from_glaciers': output_MATILDA['Q_DDM_scaled'],
                'total_runoff': output_MATILDA['Q_Total'],
                'observed_runoff': output_MATILDA['Qobs']}, index=output_MATILDA.index)

    else:
        output_MATILDA_compact = pd.DataFrame(
            {'avg_temp_catchment': output_MATILDA['HBV_temp'],
             'prec': output_MATILDA['HBV_prec'],
             'rain': output_MATILDA['HBV_rain'],
             'snow': output_MATILDA['HBV_snow'],
             'snowpack': output_MATILDA['HBV_snowpack'],
             'soil_moisture': output_MATILDA['HBV_soil_moisture'],
             'upper_groundwater': output_MATILDA['HBV_upper_gw'],
             'lower_groundwater': output_MATILDA['HBV_lower_gw'],
             'snow_melt': output_MATILDA['HBV_melt_off_glacier'],
             'total_refreezing': output_MATILDA['HBV_refreezing'],
             'actual_evaporation': output_MATILDA['HBV_AET'],
             'runoff': output_MATILDA['Q_HBV'],
             'observed_runoff': output_MATILDA['Qobs']}, index=output_MATILDA.index)

    # if obs is not None:
    #     output_MATILDA.loc[output_MATILDA.isnull().any(axis=1), :] = np.nan



    # Model efficiency coefficients
    if obs is not None:
        sim = output_MATILDA["Q_Total"]
        target = output_MATILDA["Qobs"]
        # Crop both timeseries to same periods without NAs
        sim_new = pd.DataFrame()
        sim_new['mod'] = pd.DataFrame(sim)
        sim_new['obs'] = target
        clean = sim_new.dropna()
        sim = clean['obs']
        target = clean['mod']

        if parameter.freq == "D":
            nash_sut = he.nse(sim, target, remove_zero=True)
            kge = he.kge_2012(sim, target, remove_zero=True)
            rmse = he.rmse(sim, target)
            mare = hydroeval.evaluator(hydroeval.mare, sim, target)
            print("*-------------------*")
            print("KGE coefficient: " + str(round(float(kge), 2)))
            print("NSE coefficient: " + str(round(nash_sut, 2)))
            print("RMSE: " + str(round(float(rmse), 2)))
            print("MARE coefficient: " + str(round(float(mare), 2)))
            print("*-------------------*")
        else:
            sim = sim.resample(parameter.freq).agg(pd.Series.sum, min_count=1)
            target = target.resample(parameter.freq).agg(pd.Series.sum, min_count=1)
            nash_sut = he.nse(sim, target, remove_zero=True)
            kge = he.kge_2012(sim, target, remove_zero=True)
            rmse = he.rmse(sim, target)
            mare = hydroeval.evaluator(hydroeval.mare, sim, target)
            print("*-------------------*")
            print("** Model efficiency based on " + parameter.freq_long + " aggregates **")
            print("KGE coefficient: " + str(round(float(kge), 2)))
            print("NSE coefficient: " + str(round(nash_sut, 2)))
            print("RMSE: " + str(round(float(rmse), 2)))
            print("MARE coefficient: " + str(round(float(mare), 2)))
            print("*-------------------*")
    else:
        kge = str("No observations available to calculate model efficiency coefficients.")

    # if obs is not None:
    dat_stats = output_MATILDA_compact.copy()
    dat_stats.loc[dat_stats.isnull().any(axis=1), :] = np.nan
    stats = create_statistics(dat_stats)

    print(stats)
    print("End of the MATILDA simulation")
    print("---")
    output_MATILDA = output_MATILDA.round(3)
    output_all = [output_MATILDA_compact, output_MATILDA, kge, stats, lookup_table, glacier_change]

    return output_all


def matilda_plots(output_MATILDA, parameter, plot_type="print"):
    """ MATILDA plotting function to plot input data, runoff output, and HBV parameters."""

    # resampling the output to the specified frequency
    def plot_data(output_MATILDA, parameter):
        if "observed_runoff" in output_MATILDA[0].columns:
            # obs = output_MATILDA[1]["Qobs"].resample(parameter.freq).agg(pd.DataFrame.sum, skipna=False)
            obs = output_MATILDA[0]["observed_runoff"].resample(parameter.freq).agg(pd.Series.sum, min_count=1)
        if "Q_DDM" in output_MATILDA[1].columns:
            plot_data = output_MATILDA[0].resample(parameter.freq).agg(
                {"avg_temp_catchment": "mean",
                 "prec_off_glaciers": "sum",
                 "prec_on_glaciers": "sum",
                 "total_precipitation": "sum",
                 "evap_off_glaciers": "sum",
                 "melt_off_glaciers": "sum",
                 "melt_on_glaciers": "sum",
                 "runoff_without_glaciers": "sum",
                 "runoff_from_glaciers": "sum",
                 "total_runoff": "sum",
                 "actual_evaporation": "sum",
                 "snowpack_off_glaciers": "mean",
                 "refreezing_snow":"sum",
                 "refreezing_ice": "sum",
                 "total_refreezing": "sum",
                 "soil_moisture": "mean",
                 "upper_groundwater": "mean",
                 "lower_groundwater": "mean"}, skipna=False)
        else:
            plot_data = output_MATILDA[0].resample(parameter.freq).agg(
                {"avg_temp_catchment": "mean",
                 "prec_off_glaciers": "sum",
                 "total_precipitation": "sum",
                 "evap_off_glaciers": "sum",
                 "runoff_without_glaciers": "sum",
                 "total_runoff": "sum",
                 "actual_evaporation": "sum",
                 "snowpack_off_glaciers": "mean",
                 "total_refreezing": "sum",
                 "soil_moisture": "mean",
                 "upper_groundwater": "mean",
                 "lower_groundwater": "mean"}, skipna=False)
        if "observed_runoff" in output_MATILDA[0].columns:
            plot_data["observed_runoff"] = obs

        plot_annual_data = output_MATILDA[0].copy()
        plot_annual_data["month"] = plot_annual_data.index.month
        plot_annual_data["day"] = plot_annual_data.index.day
        plot_annual_data = plot_annual_data.groupby(["month", "day"]).mean()
        plot_annual_data["date"] = pd.date_range(parameter.sim_start, freq='D', periods=len(plot_annual_data)).strftime('%Y-%m-%d')
        plot_annual_data = plot_annual_data.set_index(plot_annual_data["date"])
        plot_annual_data.index = pd.to_datetime(plot_annual_data.index)
        plot_annual_data["plot"] = 0
        if parameter.freq == "Y":
            plot_annual_data = plot_annual_data.resample("M").agg(pd.Series.sum, min_count=1)
        else:
            plot_annual_data = plot_annual_data.resample(parameter.freq).agg(pd.Series.sum, min_count=1)

        return plot_data, plot_annual_data

    # Plotting the meteorological parameters
    def plot_meteo(plot_data, parameter):
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 6))

        x_vals = plot_data.index.to_pydatetime()
        plot_length = len(x_vals)
        ax1.plot(x_vals, (plot_data["avg_temp_catchment"]), c="#d7191c")
        if plot_length > (365 * 5):
            # bar chart has very poor performance for large data sets -> switch to line chart
            ax2.fill_between(x_vals, plot_data["total_precipitation"], plot_data["prec_off_glaciers"], color='#77bbff')
            ax2.fill_between(x_vals, plot_data["prec_off_glaciers"], 0, color='#3594dc')
        else:
            ax2.bar(x_vals, plot_data["prec_off_glaciers"], width=10, color="#3594dc")
            ax2.bar(x_vals, plot_data["prec_on_glaciers"], width=10, color="#77bbff", bottom=plot_data["prec_off_glaciers"])
        ax3.plot(x_vals, plot_data["evap_off_glaciers"], c="#008837")
        plt.xlabel("Date", fontsize=9)
        ax1.grid(linewidth=0.25), ax2.grid(linewidth=0.25), ax3.grid(linewidth=0.25)
        ax3.sharey(ax2)
        ax1.set_title("Mean temperature", fontsize=9)
        ax2.set_title("Precipitation off/on glacier", fontsize=9)
        ax3.set_title("Pot. evapotranspiration", fontsize=9)
        ax1.set_ylabel("[°C]", fontsize=9)
        ax2.set_ylabel("[mm]", fontsize=9)
        ax3.set_ylabel("[mm]", fontsize=9)
        if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
            fig.suptitle(
                parameter.freq_long + " meteorological input parameters in " + str(plot_data.index.values[-1])[
                                                                               :4],
                size=14)
        else:
            fig.suptitle(
                parameter.freq_long + " meteorological input parameters in " + str(plot_data.index.values[0])[
                                                                               :4] + "-" + str(
                    plot_data.index.values[-1])[:4], size=14)
        plt.tight_layout()
        fig.set_size_inches(10, 6)
        return fig

    def plot_runoff(plot_data, plot_annual_data, parameter):
        plot_data["plot"] = 0
        # plot_data.loc[plot_data.isnull().any(axis=1), :] = np.nan
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [2.75, 1]})

        x_vals = plot_data.index.to_pydatetime()
        if 'observed_runoff' in plot_data.columns:
            ax1.plot(x_vals, plot_data["observed_runoff"], c="#E69F00", label="", linewidth=1.2)
        ax1.fill_between(x_vals, plot_data["plot"], plot_data["runoff_without_glaciers"], color='#56B4E9',
                         alpha=.75, label="")
        if "total_runoff" in plot_data.columns:
            ax1.plot(x_vals, plot_data["total_runoff"], c="k", label="", linewidth=0.75, alpha=0.75)
            ax1.fill_between(x_vals, plot_data["runoff_without_glaciers"], plot_data["total_runoff"], color='#CC79A7',
                             alpha=.75, label="")
        ax1.set_ylabel("Runoff [mm]", fontsize=9)
        if isinstance(output_MATILDA[2], float):
            anchored_text = AnchoredText('KGE coeff ' + str(round(output_MATILDA[2], 2)), loc=1, frameon=False)
        elif 'observed_runoff' not in plot_data.columns:
            anchored_text = AnchoredText(' ', loc=2, frameon=False)
        else:
            anchored_text = AnchoredText('KGE coeff exceeds boundaries', loc=2, frameon=False)
        ax1.add_artist(anchored_text)

        x_vals = plot_annual_data.index.to_pydatetime()
        if 'observed_runoff' in plot_annual_data.columns:
            ax2.plot(x_vals, plot_annual_data["observed_runoff"], c="#E69F00",
                     label="Observations", linewidth=1.2)
        ax2.fill_between(x_vals, plot_annual_data["plot"], plot_annual_data["runoff_without_glaciers"], color='#56B4E9',
                         alpha=.75, label="MATILDA catchment runoff")
        if "total_runoff" in plot_annual_data.columns:
            ax2.plot(x_vals, plot_annual_data["total_runoff"], c="k", label="MATILDA total runoff",
                     linewidth=0.75, alpha=0.75)
            ax2.fill_between(x_vals, plot_annual_data["runoff_without_glaciers"], plot_annual_data["total_runoff"], color='#CC79A7',
                             alpha=.75, label="MATILDA glacial runoff")
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        ax2.set_ylabel("Runoff [mm]", fontsize=9)
        if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
            plt.suptitle(
                parameter.freq_long + " MATILDA simulation for the period " + str(plot_data.index.values[-1])[:4],
                size=14)
        else:
            plt.suptitle(parameter.freq_long + " MATILDA simulation for the period " + str(plot_data.index.values[0])[
                                                                                       :4] + "-" + str(
                plot_data.index.values[-1])[:4], size=14)
        handles, labels = ax2.get_legend_handles_labels()
        fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
                   bbox_transform=plt.gcf().transFigure)
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.12)
        return fig

    # Plotting the HBV output parameters
    def plot_hbv(plot_data, parameter):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(10, 6))

        x_vals = plot_data.index.to_pydatetime()
        ax1.plot(x_vals, plot_data["actual_evaporation"], "k")
        ax2.plot(x_vals, plot_data["soil_moisture"], "k")
        ax3.plot(x_vals, plot_data["snowpack_off_glaciers"], "k")
        ax4.plot(x_vals, plot_data["upper_groundwater"], "k")
        ax5.plot(x_vals, plot_data["lower_groundwater"], "k")
        ax1.set_title("Actual evapotranspiration", fontsize=9)
        ax2.set_title("Soil moisture", fontsize=9)
        ax3.set_title("Water in snowpack", fontsize=9)
        ax4.set_title("Upper groundwater box", fontsize=9)
        ax5.set_title("Lower groundwater box", fontsize=9)
        plt.xlabel("Date", fontsize=9)
        ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9)
        ax4.set_ylabel("[mm]", fontsize=9), ax5.set_ylabel("[mm]", fontsize=9)
        if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
            fig.suptitle(parameter.freq_long + " output from the HBV model in the period " + str(
                plot_data.index.values[-1])[:4],
                         size=14)
        else:
            fig.suptitle(parameter.freq_long + " output from the HBV model in the period " + str(
                plot_data.index.values[0])[
                                                                                             :4] + "-" + str(
                plot_data.index.values[-1])[:4], size=14)
        plt.tight_layout()
        fig.set_size_inches(10, 6)
        return fig

    # Plotting the meteorological parameters with Plotly
    def plot_plotly_meteo(plot_data, fig, row):
        x_vals = plot_data.index.to_pydatetime()
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["avg_temp_catchment"], name="Mean temperature", line_color="#d7191c", legendgroup="meteo",
                       legendgrouptitle_text="Meteo"),
            row=row, col=1, secondary_y=False)
        # fig.add_trace(
        #     go.Bar(x=x_vals, y=plot_data["total_precipitation"], name="Precipitation sum", marker_color="#2c7bb6",
        #                legendgroup="meteo",  offsetgroup=0),
        #     row=row, col=1, secondary_y=True)
        fig.add_trace(
            go.Bar(x=x_vals, y=plot_data["prec_off_glaciers"], name="Precipitation off glacier", marker_color="#3594dc",
                       legendgroup="meteo"),
            row=row, col=1, secondary_y=True)
        fig.add_trace(
            go.Bar(x=x_vals, y=plot_data["prec_on_glaciers"], name="Precipitation on glacier", marker_color="#77bbff",
                       legendgroup="meteo"),
            row=row, col=1, secondary_y=True)
        fig.add_trace(
            go.Bar(x=x_vals, y=plot_data["evap_off_glaciers"] * -1, name="Pot. evapotranspiration", marker_color="#008837",
                       legendgroup="meteo"),
            row=row, col=1, secondary_y=True)

    # Plotting the runoff/refreezing output parameters with Plotly
    def plot_plotly_runoff(plot_data, fig, row):
        x_vals = plot_data.index.to_pydatetime()
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["runoff_without_glaciers"], name="MATILDA catchment runoff", fillcolor="#5893D4",
                       legendgroup="runoff", legendgrouptitle_text="Runoff comparison", stackgroup='one', mode='none'),
            row=row, col=1)

        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["runoff_from_glaciers"], name="MATILDA glacial runoff (stacked)",
                       fillcolor="#CC79A7",
                       legendgroup="runoff", stackgroup='one', mode='none'),
            row=row, col=1)

        if 'observed_runoff' in plot_data.columns:
            fig.add_trace(
                go.Scatter(x=x_vals, y=plot_data["observed_runoff"], name="Observations", line_color="#E69F00",
                           legendgroup="runoff"),
                row=row, col=1)
        if 'total_runoff' in plot_data.columns:
            fig.add_trace(
                go.Scatter(x=x_vals, y=plot_data["total_runoff"], name="MATILDA total runoff", line_color="black",
                           legendgroup="runoff"),
                row=row, col=1)

        # two new series for refreezing
        if 'refreezing_snow' in plot_data.columns:
            fig.add_trace(
                go.Scatter(x=x_vals, y=plot_data["refreezing_snow"], name="MATILDA snow refreezing",
                           fillcolor="#adb5bd", legendgroup="refreeze", legendgrouptitle_text="Refreezing",
                           mode='none', fill='tozeroy'),
                row=row, col=1)
        if 'refreezing_ice' in plot_data.columns:
            fig.add_trace(
                go.Scatter(x=x_vals, y=plot_data["refreezing_ice"], name="MATILDA ice refreezing",
                           fillcolor="#6c757d", legendgroup="refreeze",
                           mode='none', fill='tozeroy'),
                row=row, col=1)

        # add coefficient to plot
        fig.add_annotation(xref='x domain', yref='y domain', x=0.99, y=0.95, xanchor="right", showarrow=False,
                           text='<b>KGE coeff ' + str(round(output_MATILDA[2], 2)) + '</b>',
                           row=row, col=1)

    # Plotting the HBV output parameters with Plotly
    def plot_plotly_hbv(plot_data, fig, row):
        x_vals = plot_data.index.to_pydatetime()
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["actual_evaporation"], name="Actual evapotranspiration", line_color='#16425b',
                       legendgroup="hbv", legendgrouptitle_text="HBV subdomains"),
            row=row, col=1)
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["soil_moisture"], name="Soil moisture", line_color='#d9dcd6',
                       legendgroup="hbv"),
            row=row, col=1)
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["snowpack_off_glaciers"], name="Water in snowpack", line_color='#81c3d7',
                       legendgroup="hbv"),
            row=row, col=1)
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["upper_groundwater"], name="Upper groundwater box", line_color='#3a7ca5',
                       legendgroup="hbv"),
            row=row, col=1)
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["lower_groundwater"], name="Lower groundwater box", line_color='#2f6690',
                       legendgroup="hbv"),
            row=row, col=1)

    def plot_plotly_runoff_contrib(plot_data, fig, row):
        x_vals = plot_data.index.to_pydatetime()
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["melt_off_glaciers"], name="Melt off glacier", fillcolor='#33193f',
                       legendgroup="runoff2", legendgrouptitle_text="Runoff contributions", stackgroup='one', mode='none'),
            row=row, col=1)
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["melt_on_glaciers"], name="Melt on glacier", fillcolor='#6c1e58',
                       legendgroup="runoff2", stackgroup='one', mode='none'),
            row=row, col=1)
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["prec_off_glaciers"], name="Precipitation off glacier", fillcolor='#a6135a',
                       legendgroup="runoff2", stackgroup='one', mode='none'),
            row=row, col=1)
        fig.add_trace(
            go.Scatter(x=x_vals, y=plot_data["prec_on_glaciers"], name="Precipitation on glacier", fillcolor='#d72f41',
                       legendgroup="runoff2", stackgroup='one', mode='none'),
            row=row, col=1)

    def plot_plotly(plot_data, plot_annual_data, parameter):
        # construct date range for chart titles
        range_from = str(plot_data.index.values[1])[:4]
        range_to = str(plot_data.index.values[-1])[:4]
        if range_from == range_to:
            date_range = range_from
        else:
            date_range = range_from + "-" + range_to
        title = [" Meteorological forcing data ",
                 " Simulated vs observed runoff ",
                 " Runoff contributions ",
                 " HBV subdomains "
                 ]
        title_f = []
        for i in range(len(title)):
            title_f.append('<b>' + title[i] + '</b>')

        # -- Plot 1 (combined charts) -- #
        # init plot
        fig1 = make_subplots(
            rows=4, cols=1, subplot_titles=title_f, shared_xaxes=True,
            vertical_spacing=0.15,
            specs=[[{"secondary_y": True}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}],
                   [{"secondary_y": False}]]
        )

        # Add subplot: METEO
        plot_plotly_meteo(plot_data, fig1, 1)

        # Add subplot: MATILDA RUNOFF
        plot_plotly_runoff(plot_data, fig1, 2)

        # Add subplot: RUNOFF CONTRIBUTION
        plot_plotly_runoff_contrib(plot_data, fig1, 3)

        # Add subplot: HBV
        plot_plotly_hbv(plot_data, fig1, 4)

        # update general layout settings
        fig1.update_layout(
            plot_bgcolor='white',
            legend=dict(groupclick="toggleitem"),
            legend_tracegroupgap=20,
            xaxis_showticklabels=True,
            xaxis2_showticklabels=True,
            xaxis3_showticklabels=True,
            barmode='relative',
            hovermode="x",
            title={
                "text": parameter.freq_long + " MATILDA Results (" + date_range + ")",
                "font_size":30,
                "x":0.5,
                "xanchor": "center"
            },
            yaxis={
                "ticksuffix": " °C"
            },
            yaxis2={
                "ticksuffix": " mm",
            },
            yaxis3={
                "ticksuffix": " mm",
                "side": "right"
            },
            yaxis4={
                "ticksuffix": " mm",
                "side": "right"
            },
            yaxis5={
                "ticksuffix": " mm",
                "side": "right"
            }
        )

        # update x axes settings
        fig1.update_xaxes(
            ticks="outside",
            ticklabelmode="period",
            dtick="M12",
            tickcolor="black",
            tickwidth=2,
            ticklen=15,
            minor=dict(
                dtick="M1",
                ticklen=5)
        )

        # -- Plot 2 (annual mean) -- #
        title_annual = title[1:3]
        title_f = []
        for i in range(len(title_annual)):
            title_f.append('<b>' + title_annual[i] + '</b>')

        # init plot
        fig2 = make_subplots(
            rows=2, cols=1, subplot_titles=title_f, shared_xaxes=True,
            vertical_spacing=0.15
        )

        # Add subplot: MATILDA RUNOFF (annual data)
        plot_plotly_runoff(plot_annual_data, fig2, 1)

        # Add subplot: RUNOFF CONTRIBUTION (annual data)
        plot_plotly_runoff_contrib(plot_annual_data, fig2, 2)

        # update general layout settings
        fig2.update_layout(
            plot_bgcolor='white',
            legend=dict(groupclick="toggleitem"),
            legend_tracegroupgap=20,
            xaxis_showticklabels=True,
            hovermode="x",
            title={
                "text": "Annual mean MATILDA Results (" + date_range + ")",
                "font_size": 30,
                "x": 0.5,
                "xanchor": "center"
            },
            yaxis={
                "ticksuffix": " mm"
            },
            yaxis2={
                "ticksuffix": " mm"
            }
        )

        # update x axes settings
        fig2.update_xaxes(
            ticks="outside",
            ticklabelmode="period",
            dtick="M1",
            tickformat="%b",
            hoverformat="%d\n%b"
        )
        fig2.update_traces(marker={'opacity': 0 })

        return [fig1, fig2]

    plot_data, plot_annual_data = plot_data(output_MATILDA, parameter)

    if plot_type == "print":
        # matplotlib
        fig1 = plot_meteo(plot_data, parameter)
        fig2 = plot_runoff(plot_data, plot_annual_data, parameter)
        fig3 = plot_hbv(plot_data, parameter)
        output_MATILDA.extend([fig1, fig2, fig3])

    elif plot_type == "interactive":
        # plotly
        figs = plot_plotly(plot_data, plot_annual_data, parameter)
        output_MATILDA.extend(figs)

    elif plot_type == "all":
        # matplot  and plotly
        fig1 = plot_meteo(plot_data, parameter)
        fig2 = plot_runoff(plot_data, plot_annual_data, parameter)
        fig3 = plot_hbv(plot_data, parameter)
        output_MATILDA.extend([fig1, fig2, fig3])

        figs = plot_plotly(plot_data, plot_annual_data, parameter)
        output_MATILDA.extend(figs)

    else:
        print("unsupported plot type")

    return output_MATILDA


def matilda_save_output(output_MATILDA, parameter, output_path, plot_type="print"):
    """Function to save the MATILDA output to local disk."""
    if output_path[-1] == '/':
        output_path = output_path + parameter.sim_start[:4] + "_" + parameter.sim_end[:4] + "_" + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S") + "/"
    else:
        output_path = output_path + "/" + parameter.sim_start[:4] + "_" + parameter.sim_end[:4] + "_" + datetime.now().strftime(
        "%Y-%m-%d_%H-%M-%S") + "/"
    os.mkdir(output_path)  # creating the folder to save the plots

    # construct date range for chart titles
    range_from = str(output_MATILDA[1].index.values[1])[:4]
    range_to = str(output_MATILDA[1].index.values[-1])[:4]
    if range_from == range_to:
        date_range = range_from
    else:
        date_range = range_from + "-" + range_to

    print("Saving the MATILDA output to disc")
    output_MATILDA[1].to_csv(output_path + "model_output_" + date_range + ".csv")
    output_MATILDA[3].to_csv(output_path + "model_stats_" + date_range + ".csv")
    parameter.to_csv(output_path + "model_parameter.csv")

    if isinstance(output_MATILDA[5], pd.DataFrame):
        output_MATILDA[5].to_csv(output_path + "glacier_area_" + date_range + ".csv")

    if plot_type == "print":
        # save plots from matplotlib as .png files
        output_MATILDA[6].savefig(output_path + "meteorological_data_" + date_range + ".png", bbox_inches='tight', dpi=output_MATILDA[6].dpi)
        output_MATILDA[7].savefig(output_path + "model_runoff_" + date_range + ".png", dpi=output_MATILDA[7].dpi)
        output_MATILDA[8].savefig(output_path + "HBV_output_" + date_range + ".png", dpi=output_MATILDA[8].dpi)

    elif plot_type == "interactive":
        # save plots from plotly as .html file
        output_MATILDA[6].write_html(output_path + 'matilda_plots_' + date_range + '.html')
        output_MATILDA[7].write_html(output_path + 'matilda_plots_annual_' + date_range + '.html')

    elif plot_type == "all":
        # save plots from matplotlib and plotly as .png files
        output_MATILDA[6].savefig(output_path + "meteorological_data_" + date_range + ".png", bbox_inches='tight', dpi=output_MATILDA[6].dpi)
        output_MATILDA[7].savefig(output_path + "model_runoff_" + date_range + ".png", dpi=output_MATILDA[7].dpi)
        output_MATILDA[8].savefig(output_path + "HBV_output_" + date_range + ".png", dpi=output_MATILDA[8].dpi)

        # save plots from plotly as .html file
        output_MATILDA[9].write_html(output_path + 'matilda_plots_' + date_range + '.html')
        output_MATILDA[10].write_html(output_path + 'matilda_plots_annual_' + date_range + '.html')


    print("---")


def matilda_simulation(input_df, obs=None, glacier_profile=None, output=None, warn=False,
                       set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D", lat=None,
                       soi=None, area_cat=None, area_glac=None, ele_dat=None, ele_glac=None, ele_cat=None,
                       plots=True, plot_type="print", hydro_year=10, elev_rescaling=False, pfilter=0.2,
                       parameter_set=None, lr_temp=-0.006, lr_prec=0, TT_snow=0,
                       TT_diff=2, CFMAX_snow=2.8, CFMAX_rel=2, BETA=1.0, CET=0.15,
                       FC=250, K0=0.055, K1=0.055, K2=0.04, LP=0.7, MAXBAS=3.0,
                       PERC=1.5, UZL=120, PCORR=1.0, SFCF=0.7, CWH=0.1, AG=0.7, RFS=0.15):
    """Function to run the whole MATILDA simulation at once."""

    print('---')
    print('MATILDA framework')
    parameter = matilda_parameter(input_df, set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start,
                                  sim_end=sim_end, freq=freq, lat=lat, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, \
                                  ele_glac=ele_glac, ele_cat=ele_cat, hydro_year=hydro_year, parameter_set = parameter_set, lr_temp=lr_temp,
                                  lr_prec=lr_prec, TT_snow=TT_snow, soi=soi, warn=warn, pfilter=pfilter, \
                                  TT_diff=TT_diff, CFMAX_snow=CFMAX_snow, CFMAX_rel=CFMAX_rel, \
                                  BETA=BETA, CET=CET, FC=FC, K0=K0, K1=K1, K2=K2, LP=LP, \
                                  MAXBAS=MAXBAS, PERC=PERC, UZL=UZL, PCORR=PCORR, SFCF=SFCF, CWH=CWH, AG=AG, RFS=RFS)

    if parameter is None:
        return

    # Data preprocessing with the MATILDA preparation script
    if obs is None:
        df_preproc = matilda_preproc(input_df, parameter)
        # Downscaling of data if necessary and the MATILDA simulation
        if glacier_profile is not None:
            output_MATILDA = matilda_submodules(df_preproc, parameter, glacier_profile=glacier_profile,
                                                elev_rescaling=elev_rescaling)
        else:
            output_MATILDA = matilda_submodules(df_preproc, parameter)
    else:
        df_preproc, obs_preproc = matilda_preproc(input_df, parameter, obs=obs)
        # Scale data if necessary and run the MATILDA simulation
        if glacier_profile is not None:
            output_MATILDA = matilda_submodules(df_preproc, parameter, obs=obs_preproc, glacier_profile=glacier_profile,
                                                elev_rescaling=elev_rescaling)
        else:
            output_MATILDA = matilda_submodules(df_preproc, parameter, obs=obs_preproc)

    # Option to suppress plots.
    if plots:
        output_MATILDA = matilda_plots(output_MATILDA, parameter, plot_type)
    else:
        return output_MATILDA
    # Creating plot for the input (meteorological) data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3) and
    # adding them to the output
    # saving the data on disc of output path is given
    if output is not None:
        matilda_save_output(output_MATILDA, parameter, output, plot_type)

    return output_MATILDA
