# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
# Import all necessary python packages
import xarray as xr
import numpy as np
import pandas as pd
import scipy.signal as ss
from datetime import date, datetime
import os
import matplotlib.pyplot as plt

# Setting the parameter for the MATILDA simulation
def MATILDA_parameter(input_df, set_up_start=None, set_up_end=None, sim_start=None, sim_end=None, freq="D",
                      area_cat=None, area_glac=None, ele_dat=None, ele_glac=None, ele_cat=None, lr_temp=-0.006, lr_prec=0, \
                      hydro_year=10, TT_snow=0, TT_rain=2, CFMAX_snow=2.8, CFMAX_ice=5.6, CFR_snow=0.05, \
                      CFR_ice=0.05, BETA=1.0, CET=0.15, FC=250, K0=0.055, K1=0.055, K2=0.04, LP=0.7, MAXBAS=3.0, \
                      PERC=1.5, UZL=120, PCORR=1.0, SFCF=0.7, CWH=0.1):

    print("Reading in the parameter for the MATILDA simulation.")
    # Checking the parameters to set the catchment properties and simulation
    if area_cat is None:
        print("WARNING: No catchment area is specified. Please provide the catchment area in km2.")
        return
    if area_glac > area_cat:
        print("WARNING: The glacier area is bigger than the overall catchment area.")
    if ele_dat is not None and ele_cat is None:
        print("WARNING: Only the elevation for the data is given, not for the catchment.")
    if area_glac is not None or area_glac > 0:
        if ele_glac is None and ele_dat is not None:
            print("WARNING: Only the elevation for the data is given, not for glacier.")
    if hydro_year > 12 and hydro_year < 1:
        print("WARNING: The month which begins the hydrological year is out of bounds [1, 12]")

    if set_up_end is not None and sim_start is not None:
        if set_up_end > sim_start:
            print("WARNING: The set up period exceeds the start of the simulation period.")
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
        freq_long = "Yearly"
    else:
        print(
            "WARNING: Data frequency " + freq + " is not supported. Supported are D (daily), W (weekly), M (monthly) or Y (yearly).")

    # Checking the model parameters
    if 1 > BETA or BETA > 6:
        print("WARNING: The parameter BETA exceeds boundaries [1, 6].")
    if 0 > CET or CET > 0.3:
        print("WARNING: The parameter CET exceeds boundaries [0, 0.3].")
    if 50 > FC or FC > 500:
        print("WARNING: The parameter FC exceeds boundaries [50, 500].")
    if 0.01 > K0 or K0 > 0.4:
        print("WARNING: The parameter K0 exceeds boundaries [0.01, 0.4].")
    if 0.01 > K1 or K1 > 0.4:
        print("WARNING: The parameter K1 exceeds boundaries [0.01, 0.4].")
    if 0.001 > K2 or K2 > 0.15:
        print("WARNING: The parameter K2 exceeds boundaries [0.001, 0.15].")
    if 0.3 > LP or LP > 1:
        print("WARNING: The parameter LP exceeds boundaries [0.3, 1].")
    if 1 >= MAXBAS or MAXBAS > 7:
        print("WARNING: The parameter MAXBAS exceeds boundaries [2, 7]. Please choose a suitable value.")
        return
    if 0 > PERC or PERC > 3:
        print("WARNING: The parameter PERC exceeds boundaries [0, 3].")
    if 0 > UZL or UZL > 500:
        print("WARNING: The parameter UZL exceeds boundaries [0, 500].")
    if 0.5 > PCORR or PCORR > 2:
        print("WARNING: The parameter PCORR exceeds boundaries [0.5, 2].")
    if TT_snow > TT_rain:
        print("WARNING: TT_snow is higher than TT_rain.")
    if -1.5 > TT_snow or TT_snow > 2.5:
        print("WARNING: The parameter TT_snow exceeds boundaries [-1.5, 2.5].")
    if -1.5 > TT_rain or TT_rain > 2.5:
        print("WARNING: The parameter TT_rain exceeds boundaries [-1.5, 2.5].")
    if 1 > CFMAX_ice or CFMAX_ice > 10:
        print("WARNING: The parameter CFMAX_ice exceeds boundaries [1, 10].")
    if 1 > CFMAX_snow or CFMAX_snow > 10:
        print("WARNING: The parameter CFMAX_snow exceeds boundaries [1, 10].")
    if 0.4 > SFCF or SFCF > 1:
        print("WARNING: The parameter SFCF exceeds boundaries [0.4, 1].")
    if 0 > CFR_ice or CFR_ice > 0.1:
        print("WARNING: The parameter CFR_ice exceeds boundaries [0, 0.1].")
    if 0 > CFR_snow or CFR_snow > 0.1:
        print("WARNING: The parameter CFR_snow exceeds boundaries [0, 0.1].")
    if 0 > CWH or CWH > 0.2:
        print("WARNING: The parameter CWH exceeds boundaries [0, 0.2].")

    parameter = pd.Series(
        {"set_up_start": set_up_start, "set_up_end": set_up_end, "sim_start": sim_start, "sim_end": sim_end, \
         "freq": freq, "freq_long": freq_long, "area_cat": area_cat, "area_glac": area_glac, "ele_dat": ele_dat, \
         "ele_glac": ele_glac, "ele_cat": ele_cat, "hydro_year": hydro_year, "lr_temp": lr_temp, \
         "lr_prec": lr_prec, "TT_snow": TT_snow, "TT_rain": TT_rain, "CFMAX_snow": CFMAX_snow, \
         "CFMAX_ice": CFMAX_ice, "CFR_snow": CFR_snow, "CFR_ice": CFR_ice, "BETA": BETA, "CET": CET, \
         "FC": FC, "K0": K0, "K1": K1, "K2": K2, "LP": LP, "MAXBAS": MAXBAS, "PERC": PERC, "UZL": UZL, \
         "PCORR": PCORR, "SFCF": SFCF, "CWH": CWH})
    print("Parameter for the MATILDA simulation are set")
    return parameter


"""MATILDA preprocessing: here the dataframes are transformed into the needed format and the unit of the observation
data is converted from m3/s to mm per day."""

def MATILDA_preproc(input_df, parameter, obs=None):
    print("---")
    print("Reading in the data")
    print("Set up period between " + str(parameter.set_up_start) + " and " + str(parameter.set_up_end) + " to get appropriate initial values")
    print("Simulation period between " + str(parameter.sim_start) + " and " + str(parameter.sim_end))
    df_preproc = input_df.copy()
    if parameter.set_up_start > parameter.sim_start:
        print("WARNING: Spin up period starts later than the simulation period")
    if isinstance(df_preproc, xr.Dataset):
        df_preproc = input_df.sel(time=slice(parameter.set_up_start, parameter.sim_end))
    else:
        df_preproc.set_index('TIMESTAMP', inplace=True)
        df_preproc.index = pd.to_datetime(df_preproc.index)
        df_preproc = df_preproc[parameter.set_up_start: parameter.sim_end]
    if obs is not None:
        obs_preproc = obs.copy()
        obs_preproc.set_index('Date', inplace=True)
        obs_preproc.index = pd.to_datetime(obs_preproc.index)
        obs_preproc = obs_preproc[parameter.sim_start: parameter.sim_end]
        # Changing the input unit from m/3 to mm.
        obs_preproc["Qobs"] = obs_preproc["Qobs"] * 86400 / (parameter.area_cat * 1000000) * 1000
        obs_preproc = obs_preproc.resample("D").sum()
        # expanding the observation period to the whole one year, filling the NAs with 0
        idx_first = obs_preproc.index.year[1]
        idx_last = obs_preproc.index.year[-1]
        idx = pd.date_range(start=date(idx_first, 1, 1), end=date(idx_last, 12, 31), freq='D', name=obs_preproc.index.name)
        obs_preproc = obs_preproc.reindex(idx)
        obs_preproc = obs_preproc.fillna(0)

    if obs is not None:
        return df_preproc, obs_preproc
    if obs is None:
        return df_preproc


"""The main MATILDA simulation. It consists of a linear downscaling of the data if elevations for data, catchment and glacier
are given. Then the DDM and HBV model are run."""

def MATILDA_submodules(df_preproc, parameter, obs=None, glacier_profile=None):
    print('---')
    print('Starting the MATILDA simulation')
    # Downscaling of dataframe to mean catchment and glacier elevation
    def glacier_downscaling(df_preproc, parameter):
        if parameter.ele_glac is not None:
            height_diff_glacier = parameter.ele_glac - parameter.ele_dat
            input_df_glacier = df_preproc.copy()
            input_df_glacier["T2"] = np.where(input_df_glacier["T2"] <= 100, input_df_glacier["T2"] + 273.15,
                                              input_df_glacier["T2"])
            input_df_glacier["T2"] = input_df_glacier["T2"] + height_diff_glacier * float(parameter.lr_temp)
            input_df_glacier["RRR"] = input_df_glacier["RRR"] + height_diff_glacier * float(parameter.lr_prec)
        else:
            input_df_glacier = df_preproc.copy()
        if parameter.ele_cat is not None:
            height_diff_catchment = parameter.ele_cat - parameter.ele_dat
            input_df_catchment = df_preproc.copy()
            input_df_catchment["T2"] = np.where(input_df_catchment["T2"] <= 100, input_df_catchment["T2"] + 273.15,
                                                input_df_catchment["T2"])
            input_df_catchment["T2"] = input_df_catchment["T2"] + height_diff_catchment * float(parameter.lr_temp)
            input_df_catchment["RRR"] = input_df_catchment["RRR"] + height_diff_catchment * float(parameter.lr_prec)
        else:
            input_df_catchment = df_preproc.copy()
        return input_df_glacier, input_df_catchment

    if parameter.ele_dat is not None:
        input_df_glacier, input_df_catchment = glacier_downscaling(df_preproc, parameter)
    else:
        input_df_glacier = df_preproc.copy()
        input_df_catchment = df_preproc.copy()

    # Calculation of the positive degree days
    def calculate_PDD(ds):
        print("Calculating the positive degree days")
        # masking the dataset to only get the glacier area
        if isinstance(ds, xr.Dataset):
            mask = ds.MASK.values
            temp = xr.where(mask == 1, ds["T2"], np.nan)
            temp = temp.mean(dim=["lat", "lon"])
            temp = xr.where(temp >= 100, temp - 273.15, temp)  # making sure the temperature is in Celsius
            temp_min = temp.resample(time="D").min(dim="time")
            temp_max = temp.resample(time="D").max(dim="time")
            temp_mean = temp.resample(time="D").mean(dim="time")
            prec = xr.where(mask == 1, ds["RRR"], np.nan)
            prec = prec.mean(dim=["lat", "lon"])
            prec = prec.resample(time="D").sum(dim="time")
            time = temp_mean["time"]
        else:
            temp = ds["T2"]
            if temp[1] >= 100:  # making sure the temperature is in Celsius
                temp = temp - 273.15
            temp_min = temp.resample("D").min()
            temp_mean = temp.resample("D").mean()
            temp_max = temp.resample("D").max()
            prec = ds["RRR"].resample("D").sum()

        pdd_ds = xr.merge([xr.DataArray(temp_mean, name="temp_mean"), xr.DataArray(temp_min, name="temp_min"), \
                           xr.DataArray(temp_max, name="temp_max"), xr.DataArray(prec)])

        # calculate the positive degree days
        pdd_ds["pdd"] = xr.where(pdd_ds["temp_mean"] > 0, pdd_ds["temp_mean"], 0)

        return pdd_ds

    """
    Degree Day Model to calculate the accumulation, snow and ice melt and runoff rate from the glaciers.
    Model input rewritten and adjusted to our needs from the pypdd function (github.com/juseg/pypdd
    - # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>)
    """

    def calculate_glaciermelt(ds, parameter):
        print("Calculating melt with the DDM")
        temp = ds["temp_mean"]
        prec = ds["RRR"]
        pdd = ds["pdd"]

        """ pypdd.py line 311
            Compute accumulation rate from temperature and precipitation.
            The fraction of precipitation that falls as snow decreases linearly
            from one to zero between temperature thresholds defined by the
            `temp_snow` and `temp_rain` attributes.
        """
        reduced_temp = (parameter.TT_rain - temp) / (parameter.TT_rain - parameter.TT_snow)
        snowfrac = np.clip(reduced_temp, 0, 1)
        accu_rate = snowfrac * prec

        # initialize snow depth and melt rates (pypdd.py line 214)
        snow_depth = xr.zeros_like(temp)
        snow_melt_rate = xr.zeros_like(temp)
        ice_melt_rate = xr.zeros_like(temp)

        """ pypdd.py line 331
            Compute melt rates from snow precipitation and pdd sum.
            Snow melt is computed from the number of positive degree days (*pdd*)
            and the `pdd_factor_snow` model attribute. If all snow is melted and
            some energy (PDD) remains, ice melt is computed using `pdd_factor_ice`.
            *snow*: array_like
                Snow precipitation rate.
            *pdd*: array_like
                Number of positive degree days.
        """

        def melt_rates(snow, pdd):
            # compute a potential snow melt
            pot_snow_melt = parameter.CFMAX_snow * pdd
            # effective snow melt can't exceed amount of snow
            snow_melt = np.minimum(snow, pot_snow_melt)
            # ice melt is proportional to excess snow melt
            ice_melt = (pot_snow_melt - snow_melt) * parameter.CFMAX_ice / parameter.CFMAX_snow
            # return melt rates
            return (snow_melt, ice_melt)

        # compute snow depth and melt rates (pypdd.py line 219)
        for i in np.arange(len(temp)):
            if i > 0:
                snow_depth[i] = snow_depth[i - 1]
            snow_depth[i] += accu_rate[i]
            snow_melt_rate[i], ice_melt_rate[i] = melt_rates(snow_depth[i], pdd[i])
            snow_depth[i] -= snow_melt_rate[i]
        total_melt = snow_melt_rate + ice_melt_rate
        runoff_rate = total_melt - parameter.CFR_snow * snow_melt_rate \
                      - parameter.CFR_ice * ice_melt_rate
        inst_smb = accu_rate - runoff_rate

        glacier_melt = xr.merge(
            [xr.DataArray(inst_smb, name="DDM_smb"), xr.DataArray(pdd, name="pdd"), \
             xr.DataArray(accu_rate, name="DDM_accumulation_rate"),
             xr.DataArray(ice_melt_rate, name="DDM_ice_melt_rate"),
             xr.DataArray(snow_melt_rate, name="DDM_snow_melt_rate"), \
             xr.DataArray(total_melt, name="DDM_total_melt"), xr.DataArray(runoff_rate, name="Q_DDM")])
        # glacier_melt = glacier_melt.assign_coords(water_year = ds["water_year"])

        # making the final dataframe
        DDM_results = glacier_melt.to_dataframe()
        DDM_results = DDM_results.round(3)
        print("Finished running the DDM")
        return DDM_results

    input_df_glacier = input_df_glacier[parameter.sim_start:parameter.sim_end]
    if parameter.area_glac > 0:
        degreedays_ds = calculate_PDD(input_df_glacier)
        output_DDM = calculate_glaciermelt(degreedays_ds, parameter)

    """ Implementing a glacier melt routine baseon the deltaH approach based on Seibert et al. (2018) and 
    Huss and al.(2010)"""

    def create_lookup_table(glacier_profile, parameter):
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

    # calculating the new glacier area for each hydrological year
    def glacier_change(output_DDM, lookup_table, glacier_profile, parameter):
        # determining from when to when the hydrological year is
        output_DDM["water_year"] = np.where((output_DDM.index.month) >= parameter.hydro_year, output_DDM.index.year + 1,
                                            output_DDM.index.year)
        # initial smb from the glacier routine script in m w.e.
        m = sum(glacier_profile["Area"] * glacier_profile["WE"])
        initial_smb = m / 1000
        # initial area
        initial_area = glacier_profile.groupby("EleZone")["Area"].sum()
        # dataframe with the smb change per hydrological year in m w.e.
        glacier_change = pd.DataFrame({"smb": output_DDM.groupby("water_year")["DDM_smb"].sum() / 1000 * 0.9}).reset_index()  # do we have to scale this?
        # adding the changes to get the whole change in comparison to the initial area
        glacier_change["smb_sum"] = np.cumsum(glacier_change["smb"])
        # percentage of how much of the initial mass melted
        glacier_change["smb_percentage"] = round((glacier_change["smb_sum"] / initial_smb) * 100)

        glacier_change_area = pd.DataFrame({"time":"initial", "glacier_area":[parameter.area_glac]})

        output_DDM["Q_DDM_updated"] = output_DDM["Q_DDM"]
        for i in range(len(glacier_change)):
            year = glacier_change["water_year"][i]
            smb = int(-glacier_change["smb_percentage"][i])
            if smb <=99:
                # getting the right row from the lookup table depending on the smb
                area_melt = lookup_table.iloc[smb]
                # getting the new glacier area by multiplying the initial area with the area changes
                new_area = np.nansum((area_melt.values * (initial_area.values)))*parameter.area_cat
            else:
                new_area = 0
            # multiplying the output with the fraction of the new area
            glacier_change_area = glacier_change_area.append({'time': year, "glacier_area":new_area}, ignore_index=True)
            output_DDM["Q_DDM_updated"] = np.where(output_DDM["water_year"] == year, output_DDM["Q_DDM"] * (new_area / parameter.area_cat), output_DDM["Q_DDM_updated"])

        return output_DDM, glacier_change_area
    if parameter.area_glac > 0:
        if glacier_profile is not None:
            lookup_table = create_lookup_table(glacier_profile, parameter)
            output_DDM, glacier_change_area = glacier_change(output_DDM, lookup_table, glacier_profile, parameter)
        else:
            # scaling glacier melt to glacier area
            output_DDM["Q_DDM"] = output_DDM["Q_DDM"] * (parameter.area_glac / parameter.area_cat)
            lookup_table = str("No lookup table generated")
            glacier_change_area = str("No glacier changes calculated")
    else:
        lookup_table = str("No lookup table generated")
        glacier_change_area = str("No glacier changes calculated")

    """
    Compute the runoff from the catchment with the HBV model
    Python Code from the LHMP and adjusted to our needs (github.com/hydrogo/LHMP -
    Ayzel Georgy. (2016). LHMP: lumped hydrological modelling playground. Zenodo. doi: 10.5281/zenodo.59501)
    For the HBV model, evapotranspiration values are needed. These are calculated with the formula by Oudin et al. (2005)
    in the unit mm / day.
    """

    def hbv_simulation(input_df_catchment, parameter):
        print("Running the HBV model")
        # 1. new temporary dataframe from input with daily values
        if "PE" in input_df_catchment.columns:
            input_df_hbv = input_df_catchment.resample("D").agg({"T2": 'mean', "RRR": 'sum', "PE": "sum"})
        else:
            input_df_hbv = input_df_catchment.resample("D").agg({"T2": 'mean', "RRR": 'sum'})

        Temp = input_df_hbv['T2']
        if Temp[1] >= 100:  # making sure the temperature is in Celsius
            Temp = Temp - 273.15
        Prec = input_df_hbv['RRR']

        # Calculation of PE with Oudin et al. 2005
        solar_constant = (1376 * 1000000) / 86400  # from 1376 J/m2s to MJm2d
        extra_rad = 27.086217947590317
        latent_heat_flux = 2.45
        water_density = 1000
        if "PE" in input_df_catchment.columns:
            Evap = input_df_hbv["PE"]
        else:
            input_df_hbv["PE"] = np.where((Temp) + 5 > 0, ((extra_rad / (water_density * latent_heat_flux)) * \
                                                           ((Temp) + 5) / 100) * 1000, 0)
            Evap = input_df_hbv["PE"]

        # 2. Calibration period:
        # 2.1 meteorological forcing preprocessing
        Temp_cal = Temp[parameter.set_up_start:parameter.set_up_end]
        Prec_cal = Prec[parameter.set_up_start:parameter.set_up_end]
        Evap_cal = Evap[parameter.set_up_start:parameter.set_up_end]
        # overall correction factor
        Prec_cal = parameter.PCORR * Prec_cal
        # precipitation separation
        # if T < parTT: SNOW, else RAIN
        RAIN_cal = np.where(Temp_cal > parameter.TT_rain, Prec_cal, 0)
        # SNOW_cal2 = np.where(Temp_cal <= parTT, Prec_cal, 0)
        reduced_temp_cal = (parameter.TT_rain - Temp_cal) / (parameter.TT_rain - parameter.TT_snow)
        snowfrac_cal = np.clip(reduced_temp_cal, 0, 1)
        SNOW_cal = snowfrac_cal * Prec_cal
        # snow correction factor
        SNOW_cal = parameter.SFCF * SNOW_cal
        SNOW_cal =SNOW_cal * (1-(parameter.area_glac / parameter.area_cat))
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
        MELTWATER_cal = np.zeros(len(Prec_cal))
        MELTWATER_cal[0] = 0.0001
        # soil moisture box
        SM_cal = np.zeros(len(Prec_cal))
        SM_cal[0] = 0.0001
        # actual evaporation
        ETact_cal = np.zeros(len(Prec_cal))
        ETact_cal[0] = 0.0001

        # 2.3 Running model for calibration period
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
            MELTWATER_cal[t] = MELTWATER_cal[t - 1] + melt
            # snowpack after melting
            SNOWPACK_cal[t] = SNOWPACK_cal[t] - melt
            # refreezing accounting
            refreezing = parameter.CFR_snow * parameter.CFMAX_snow * (parameter.TT_snow - Temp_cal[t])
            # control refreezing
            if refreezing < 0: refreezing = 0
            refreezing = min(refreezing, MELTWATER_cal[t])
            # snowpack after refreezing
            SNOWPACK_cal[t] = SNOWPACK_cal[t] + refreezing
            # meltwater after refreezing
            MELTWATER_cal[t] = MELTWATER_cal[t] - refreezing
            # recharge to soil
            tosoil = MELTWATER_cal[t] - (parameter.CWH * SNOWPACK_cal[t]);
            # control recharge to soil
            if tosoil < 0: tosoil = 0
            # meltwater after recharge to soil
            MELTWATER_cal[t] = MELTWATER_cal[t] - tosoil

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
        print("Finished spin up for initital parameters for the HBV model")

        # 3. meteorological forcing preprocessing for simulation
        # overall correction factor
        Prec = parameter.PCORR * Prec
        # precipitation separation
        # if T < parTT: SNOW, else RAIN
        RAIN = np.where(Temp > parameter.TT_snow, Prec, 0)
        # SNOW = np.where(Temp <= parTT, Prec, 0)
        reduced_temp = (parameter.TT_rain - Temp) / (parameter.TT_rain - parameter.TT_snow)
        snowfrac = np.clip(reduced_temp, 0, 1)
        SNOW = snowfrac * Prec
        # snow correction factor
        SNOW = parameter.SFCF * SNOW
        # snow correction factor
        SNOW = parameter.SFCF * SNOW
        SNOW = SNOW * (1-(parameter.area_glac / parameter.area_cat))
        # evaporation correction
        # a. calculate long-term averages of daily temperature
        Temp_mean = np.array([Temp.loc[Temp.index.dayofyear == x].mean() \
                              for x in range(1, 367)])
        # b. correction of Evaporation daily values
        Evap = Evap.index.map(lambda x: (1 + parameter.CET * (Temp[x] - Temp_mean[x.dayofyear - 1])) * Evap[x])
        # c. control Evaporation
        Evap = np.where(Evap > 0, Evap, 0)

        # 4. initialize boxes and initial conditions after calibration
        # snowpack box
        SNOWPACK = np.zeros(len(Prec))
        SNOWPACK[0] = SNOWPACK_cal[-1]
        # meltwater box
        MELTWATER = np.zeros(len(Prec))
        MELTWATER[0] = 0.0001
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
            # day-degree simple melting
            melt = parameter.CFMAX_snow * (Temp[t] - parameter.TT_snow)
            # control melting
            if melt < 0: melt = 0
            melt = min(melt, SNOWPACK[t])
            # how meltwater box forms
            MELTWATER[t] = MELTWATER[t - 1] + melt
            # snowpack after melting
            SNOWPACK[t] = SNOWPACK[t] - melt
            # refreezing accounting
            refreezing = parameter.CFR_snow * parameter.CFMAX_snow * (parameter.TT_snow - Temp[t])
            # control refreezing
            if refreezing < 0: refreezing = 0
            refreezing = min(refreezing, MELTWATER[t])
            # snowpack after refreezing
            SNOWPACK[t] = SNOWPACK[t] + refreezing
            # meltwater after refreezing
            MELTWATER[t] = MELTWATER[t] - refreezing
            # recharge to soil
            tosoil = MELTWATER[t] - (parameter.CWH * SNOWPACK[t]);
            # control recharge to soil
            if tosoil < 0: tosoil = 0
            # meltwater after recharge to soil
            MELTWATER[t] = MELTWATER[t] - tosoil

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
            # upper groudwater box
            SUZ[t] = SUZ[t - 1] + recharge + excess
            # percolation control
            perc = min(SUZ[t], parameter.PERC)
            # update upper groudwater box
            SUZ[t] = SUZ[t] - perc
            # runoff from the highest part of upper grondwater box (surface runoff)
            Q0 = parameter.K0 * max(SUZ[t] - parameter.UZL, 0)
            # update upper groudwater box
            SUZ[t] = SUZ[t] - Q0
            # runoff from the middle part of upper groundwater box
            Q1 = parameter.K1 * SUZ[t]
            # update upper groudwater box
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
        # here are my method with simple forward filter based on Butterworht filter design
        # calculate Numerator (b) and denominator (a) polynomials of the IIR filter
        parMAXBAS = int(parameter.MAXBAS)
        b, a = ss.butter(parMAXBAS, 1 / parMAXBAS)
        # implement forward filter
        Qsim_smoothed = ss.lfilter(b, a, Qsim)
        # control smoothed runoff
        Qsim_smoothed = np.where(Qsim_smoothed > 0, Qsim_smoothed, 0)

        Qsim = Qsim_smoothed
        hbv_results = pd.DataFrame(
            {"T2": Temp, "RRR": Prec, "PE": Evap, "HBV_snowpack": SNOWPACK, "HBV_soil_moisture": SM, "HBV_AET": ETact, \
             "HBV_upper_gw": SUZ, "HBV_lower_gw": SLZ, "Q_HBV": Qsim}, index=input_df_hbv.index)
        hbv_results = hbv_results.round(3)
        print("Finished running the HBV")
        return hbv_results

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
            output_MATILDA["Q_Total"] = output_MATILDA["Q_HBV"] + output_MATILDA["Q_DDM_updated"]
        else:
            output_MATILDA["Q_Total"] = output_MATILDA["Q_HBV"] + output_MATILDA["Q_DDM"]
    else:
        output_MATILDA["Q_Total"] = output_MATILDA["Q_HBV"]

    output_MATILDA = output_MATILDA[parameter.sim_start:parameter.sim_end]

    # Nash–Sutcliffe model efficiency coefficient
    def NS(output_MATILDA, obs):
        nash_sut = 1 - np.sum((obs["Qobs"] - output_MATILDA["Q_Total"]) ** 2) / (
            np.sum((obs["Qobs"] - obs["Qobs"].mean()) ** 2))
        if nash_sut > 1 or nash_sut < -1:
            nash_sut = "error"
        return nash_sut

    if obs is not None:
        nash_sut = NS(output_MATILDA, obs)

        if nash_sut == "error":
            print("ERROR. The Nash–Sutcliffe model efficiency coefficient is outside the range of -1 to 1")
        else:
            print("The Nash–Sutcliffe model efficiency coefficient of the MATILDA run is " + str(round(nash_sut, 2)))
    if obs is None:
        nash_sut = str("No observations available to calculate the Nash–Sutcliffe model efficiency coefficient")

    def create_statistics(output_MATILDA):
        stats = output_MATILDA.describe()
        sum = pd.DataFrame(output_MATILDA.sum())
        sum.columns = ["sum"]
        sum = sum.transpose()
        stats = stats.append(sum)
        stats = stats.round(3)
        return stats

    stats = create_statistics(output_MATILDA)

    print(stats)
    print("End of the MATILDA simulation")
    print("---")
    output_all = [output_MATILDA, nash_sut, stats, lookup_table, glacier_change_area]

    return output_all


""" MATILDA plotting function to plot the input data, runoff output and HBV parameters."""


def MATILDA_plots(output_MATILDA, parameter):
    # resampling the output to the specified frequency
    def plot_data(output_MATILDA, parameter):
        if "Qobs" in output_MATILDA[0].columns:
            obs = output_MATILDA[0]["Qobs"].resample(parameter.freq).agg(pd.DataFrame.sum, skipna=False)
        if "Q_DDM" in output_MATILDA[0].columns:
            plot_data = output_MATILDA[0].resample(parameter.freq).agg(
                {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", \
                "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
                 "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})
        else:
            plot_data = output_MATILDA[0].resample(parameter.freq).agg(
                {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", \
                  "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
                 "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})
        if "Qobs" in output_MATILDA[0].columns:
            plot_data["Qobs"] = obs
        return plot_data

    # Plotting the meteorological parameters
    def plot_meteo(plot_data, parameter):
        fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10, 6))
        ax1.plot(plot_data.index.to_pydatetime(), (plot_data["T2"]), c="#d7191c")
        ax2.bar(plot_data.index.to_pydatetime(), plot_data["RRR"], width=10, color="#2c7bb6")
        ax3.plot(plot_data.index.to_pydatetime(), plot_data["PE"], c="#008837")
        plt.xlabel("Date", fontsize=9)
        ax1.grid(linewidth=0.25), ax2.grid(linewidth=0.25), ax3.grid(linewidth=0.25)
        ax1.set_title("Mean temperature", fontsize=9)
        ax2.set_title("Precipitation sum", fontsize=9)
        ax3.set_title("Evapotranspiration sum", fontsize=9)
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
                parameter.freq_long + " meteorological input parameters in " + str(plot_data.index.values[1])[
                                                                               :4] + "-" + str(
                    plot_data.index.values[-1])[:4], size=14)
        plt.tight_layout()
        fig.set_size_inches(10, 6)
        return fig

    def plot_runoff(plot_data, parameter):
        plot_data["plot"] = 0
        fig = plt.figure(figsize=(10, 6))
        if 'Qobs' in plot_data.columns:
            plt.plot(plot_data.index.to_pydatetime(), plot_data["Qobs"], c="#E69F00",
                     label="Observations", linewidth=1.2)
        plt.fill_between(plot_data.index.to_pydatetime(), plot_data["plot"], plot_data["Q_HBV"], color='#56B4E9',
                         alpha=.75, label="MATILDA catchment runoff")
        if "Q_DDM" in plot_data.columns:
            plt.plot(plot_data.index.to_pydatetime(), plot_data["Q_Total"], c="k", label="MATILDA total runoff",
                     linewidth=0.75, alpha=0.75)
            plt.fill_between(plot_data.index.to_pydatetime(), plot_data["Q_HBV"], plot_data["Q_Total"],
                            color='#CC79A7', alpha=.75, label="MATILDA glacial runoff")
        plt.legend()
        plt.ylabel("Runoff [mm]", fontsize=9)
        if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
            plt.title(
                parameter.freq_long + " MATILDA simulation for the period " + str(plot_data.index.values[-1])[
                                                                              :4],
                size=14)
        else:
            plt.title(
                parameter.freq_long + " MATILDA simulation for the period " + str(plot_data.index.values[1])[
                                                                              :4] + "-" + str(
                    plot_data.index.values[-1])[:4], size=14)
        if output_MATILDA[1] == "error":
            plt.text(0.77, 0.9, 'NS coeff exceeds boundaries', fontsize=8, transform=fig.transFigure)
        elif isinstance(output_MATILDA[1], float):
            plt.text(0.85, 0.9, 'NS coeff ' + str(round(output_MATILDA[1], 2)), fontsize=8, transform=fig.transFigure)
        plt.tight_layout()
        fig.set_size_inches(10, 6)
        return fig

    # Plotting the HBV output parameters
    def plot_hbv(plot_data, parameter):
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(10, 6))
        ax1.plot(plot_data.index.to_pydatetime(), plot_data["HBV_AET"], "k")
        ax2.plot(plot_data.index.to_pydatetime(), plot_data["HBV_soil_moisture"], "k")
        ax3.plot(plot_data.index.to_pydatetime(), plot_data["HBV_snowpack"], "k")
        ax4.plot(plot_data.index.to_pydatetime(), plot_data["HBV_upper_gw"], "k")
        ax5.plot(plot_data.index.to_pydatetime(), plot_data["HBV_lower_gw"], "k")
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
                plot_data.index.values[1])[
                                                                                             :4] + "-" + str(
                plot_data.index.values[-1])[:4], size=14)
        plt.tight_layout()
        fig.set_size_inches(10, 6)
        return fig

    plot_data = plot_data(output_MATILDA, parameter)
    fig1 = plot_meteo(plot_data, parameter)
    fig2 = plot_runoff(plot_data, parameter)
    fig3 = plot_hbv(plot_data, parameter)
    output_MATILDA.extend([fig1, fig2, fig3])
    return output_MATILDA


# Function to save the MATILDA output to the local machine.
def MATILDA_save_output(output_MATILDA, parameter, output_path):
    output_path = output_path + parameter.sim_start[:4] + "_" + parameter.sim_end[:4] + "_" + datetime.now().strftime(
        "%Y-%m-%d_%H:%M:%S") + "/"
    os.mkdir(output_path)  # creating the folder to save the plots

    print("Saving the MATILDA output to disc")
    output_MATILDA[0].to_csv(output_path + "model_output_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
        output_MATILDA[0].index.values[-1])[:4] + ".csv")
    output_MATILDA[2].to_csv(output_path + "model_stats_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
        output_MATILDA[0].index.values[-1])[:4] + ".csv")
    parameter.to_csv(output_path + "model_parameter.csv")

    if str(output_MATILDA[0].index.values[1])[:4] == str(output_MATILDA[0].index.values[-1])[:4]:
        output_MATILDA[5].savefig(
            output_path + "meteorological_data_" + str(output_MATILDA[0].index.values[-1])[:4] + ".png",
            dpi=output_MATILDA[5].dpi)
    else:
        output_MATILDA[5].savefig(
            output_path + "meteorological_data_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
                output_MATILDA[0].index.values[-1])[:4] + ".png", dpi=output_MATILDA[5].dpi)

    if str(output_MATILDA[0].index.values[1])[:4] == str(output_MATILDA[0].index.values[-1])[:4]:
        output_MATILDA[6].savefig(output_path + "model_runoff_" + str(output_MATILDA[0].index.values[-1])[:4] + ".png",
                                  dpi=output_MATILDA[6].dpi)
    else:
        output_MATILDA[6].savefig(
            output_path + "model_runoff_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
                output_MATILDA[0].index.values[-1])[:4] + ".png",
            dpi=output_MATILDA[6].dpi)

    if str(output_MATILDA[0].index.values[1])[:4] == str(output_MATILDA[0].index.values[-1])[:4]:
        output_MATILDA[7].savefig(output_path + "HBV_output_" + str(output_MATILDA[0].index.values[-1])[:4] + ".png",
                                  dpi=output_MATILDA[7].dpi)
    else:
        output_MATILDA[7].savefig(
            output_path + "HBV_output_" + str(output_MATILDA[0].index.values[1])[:4] + "-" + str(
                output_MATILDA[0].index.values[-1])[:4] + ".png",
            dpi=output_MATILDA[7].dpi)
    print("---")


"""Function to run the whole MATILDA simulation in one function. """


def MATILDA_simulation(input_df, obs=None, glacier_profile=None, output=None, set_up_start=None, set_up_end=None, \
                       sim_start=None, sim_end=None, freq="D", area_cat=None, area_glac=None, ele_dat=None, ele_glac=None, \
                       ele_cat=None, hydro_year=10, lr_temp=-0.006, lr_prec=0, TT_snow=0, TT_rain=2, \
                       CFMAX_snow=2.8, CFMAX_ice=5.6, CFR_snow=0.05, CFR_ice=0.05, BETA=1.0, CET=0.15, FC=250, \
                       K0=0.055, K1=0.055, K2=0.04, LP=0.7, MAXBAS=3.0, PERC=1.5, UZL=120, PCORR=1.0, SFCF=0.7,
                       CWH=0.1):
    print('---')
    print('MATILDA framework')
    parameter = MATILDA_parameter(input_df, set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start,
                                  sim_end=sim_end, \
                                  freq=freq, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, \
                                  ele_glac=ele_glac, ele_cat=ele_cat, hydro_year=hydro_year, lr_temp=lr_temp,
                                  lr_prec=lr_prec, TT_snow=TT_snow, \
                                  TT_rain=TT_rain, CFMAX_snow=CFMAX_snow, CFMAX_ice=CFMAX_ice, CFR_snow=CFR_snow, \
                                  CFR_ice=CFR_ice, BETA=BETA, CET=CET, FC=FC, K0=K0, K1=K1, K2=K2, LP=LP, \
                                  MAXBAS=MAXBAS, PERC=PERC, UZL=UZL, PCORR=PCORR, SFCF=SFCF, CWH=CWH)

    if parameter is None:
        return

    # Data preprocessing with the MATILDA preparation script
    if obs is None:
        df_preproc = MATILDA_preproc(input_df, parameter)
        # Downscaling of data if necessary and the MATILDA simulation
        if glacier_profile is not None:
            output_MATILDA = MATILDA_submodules(df_preproc, parameter, glacier_profile=glacier_profile)
        else:
            output_MATILDA = MATILDA_submodules(df_preproc, parameter)
    else:
        df_preproc, obs_preproc = MATILDA_preproc(input_df, parameter, obs=obs)
        # Downscaling of data if necessary and the MATILDA simulation
        if glacier_profile is not None:
            output_MATILDA = MATILDA_submodules(df_preproc, parameter, obs=obs_preproc, glacier_profile=glacier_profile)
        else:
            output_MATILDA = MATILDA_submodules(df_preproc, parameter, obs=obs_preproc)

    output_MATILDA = MATILDA_plots(output_MATILDA, parameter)
    # Creating plot for the input (meteorological) data (fig1), MATILDA runoff simulation (fig2) and HBV variables (fig3) and
    # adding them to the output

    # saving the data on disc of output path is given
    if output is not None:
        MATILDA_save_output(output_MATILDA, parameter, output)

    return output_MATILDA
