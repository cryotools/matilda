# -*- coding: UTF-8 -*-
"""
MATILDA: Modeling wATer resources In gLacierizeD cAtchments
===========================================================

Description
-----------
MATILDA is a hydrological modeling framework designed to simulate runoff contributions in glacierized catchments. 
It integrates a degree-day model (DDM) with the HBV hydrological model (Bergström, 1976) to partition runoff into 
glacial and non-glacial components. The model can account for changes in glacier geometry through annual rescaling 
based on elevation-band profiles.

Core features include:
- Preprocessing of meteorological input data.
- Degree-day-based glacier melt modeling.
- HBV-based rainfall-runoff simulation.
- Optional elevation-based glacier geometry rescaling.
- Output visualization and statistics generation.

References
----------
1. Bergström, S. (1976). Development and application of a conceptual runoff model for Scandinavian catchments. 
   SMHI Reports RHO No. 7.
2. Ayzel, G. (2016). Lumped Hydrological Models Playground ([LHMP](https://github.com/hydrogo/LHMP)). Zenodo. 
   https://doi.org/10.5281/zenodo.59501
3. Seguinot, J. (2013–2018). Python positive degree-day model for glacier surface mass balance ([pypdd](https://github.com/juseg/pypdd)).
4. Oudin, L., et al. (2005). Which potential evapotranspiration input for a lumped rainfall-runoff model?: 
   Part 2—Towards a simple and efficient potential evapotranspiration model for rainfall-runoff modeling. 
   Journal of Hydrology, 303(1), 290–306. https://doi.org/10.1016/j.jhydrol.2004.08.026

Dependencies
------------
- Python 3.x
- pandas
- numpy
- matplotlib
- xarray
- hydroeval
- scipy

Usage
-----
Run the MATILDA framework using the `matilda_simulation` function, which combines all preprocessing, modeling, and postprocessing steps. 
The framework allows customization via parameters, input datasets, and optional outputs (e.g., plots and CSV files).

Example:
    ```python
    output = matilda_simulation(
        input_df=your_data,
        obs=observed_runoff,
        glacier_profile=glacier_profile_data,
        output="output_folder",
        plots=True
    )
    ```

License
-------
This software is released under the MIT License. See LICENSE file for details.

Contact
-------
For questions or contributions, please contact:
- Developer: Phillip Schuster
- Email: phillip.schuster@geo.hu-berlin.de
- Institution: Humboldt-Universität zu Berlin
"""

import os
from datetime import date, datetime
import warnings
import copy
import importlib.resources
import json
import pandas as pd
import numpy as np
import xarray as xr
import scipy.signal as ss
import hydroeval
import HydroErr as he
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
from matplotlib.offsetbox import AnchoredText
from plotly.subplots import make_subplots

warnings.filterwarnings(action="ignore", module="HydroErr")


def matilda_parameter(
    input_df,
    set_up_start=None,
    set_up_end=None,
    sim_start=None,
    sim_end=None,
    freq="D",
    lat=None,
    area_cat=None,
    area_glac=None,
    ele_dat=None,
    ele_glac=None,
    ele_cat=None,
    warn=False,
    **matilda_param,
):
    """
    Initialize parameters for the MATILDA simulation.

    This function processes user-defined and default parameters for the MATILDA model,
    validates parameter ranges, calculates derived parameters, and prepares the final
    parameter set for the simulation.

    Parameters:
    -----------
    input_df : pandas.DataFrame
        Input data containing meteorological and optional discharge data.
    set_up_start : str or None, optional
        Start date for the model setup period (default: inferred from input data).
    set_up_end : str or None, optional
        End date for the model setup period (default: one year after `set_up_start`).
    sim_start : str or None, optional
        Start date for the simulation period (default: inferred from input data).
    sim_end : str or None, optional
        End date for the simulation period (default: inferred from input data).
    freq : str, optional
        Resampling frequency for output data. Options: 'D' (daily), 'W' (weekly),
        'M' (monthly), 'Y' (yearly). Default is 'D'.
    lat : float, required
        Latitude of the study area in degrees, required for potential evapotranspiration calculations.
    area_cat : float, required
        Total catchment area in square kilometers.
    area_glac : float or None, optional
        Glacierized area within the catchment in square kilometers (default: 0).
    ele_dat : float or None, optional
        Reference elevation for input meteorological data (default: None).
    ele_glac : float or None, optional
        Mean elevation of the glacierized area (default: None).
    ele_cat : float or None, optional
        Mean elevation of the entire catchment (default: None).
    warn : bool, optional
        Whether to display warnings for potential configuration issues (default: False).
    **matilda_param : dict, optional
        Additional model parameters, passed as key-value pairs. If a parameter is not
        provided, default values from the parameter JSON file are used.

    Returns:
    --------
    pandas.Series
        A series containing all configured parameters for the MATILDA simulation.

    Raises:
    -------
    ValueError
        If required parameters (`lat` or `area_cat`) are not provided or if parameter
        values are outside acceptable bounds.

    Notes:
    ------
    - Derived parameters such as `TT_rain` (threshold temperature for rain) and
      `CFMAX_ice` (melt factor for ice) are computed automatically.
    - Reference elevations (`ele_cat`, `ele_glac`) are used to scale meteorological input data.
    - The parameter set is validated against predefined bounds from a JSON configuration file.
    - Default values are stored in the `parameters.json` file.
    """

    # Filter warnings:
    if not warn:
        warnings.filterwarnings(action="ignore")

    print("Reading parameters for MATILDA simulation")

    # Parameter checks
    if lat is None:
        raise ValueError(
            "No latitude specified. Please provide 'lat' to calculate potential evapotranspiration (PE)."
        )
    if area_cat is None:
        raise ValueError(
            "No catchment area specified. Please provide 'area_cat' in km²."
        )
    if area_glac is None:
        area_glac = 0
    if area_glac > area_cat:
        raise ValueError(
            "Glacier area ('area_glac') exceeds overall catchment area ('area_cat')."
        )
    if ele_dat is not None and ele_cat is None:
        print(
            "WARNING: Catchment reference elevation is missing. The data cannot be elevation scaled."
        )
    if ele_cat is None or ele_glac is None:
        print(
            "WARNING: Reference elevations for catchment and glacier area need to be provided to scale the model"
            "domains correctly!"
        )
        ele_non_glac = None
    else:
        # Calculate the mean elevation of the non-glacierized catchment area
        ele_non_glac = (
            (ele_cat - area_glac / area_cat * ele_glac)
            * area_cat
            / (area_cat - area_glac)
        )
    if area_glac is not None or area_glac > 0:
        if ele_glac is None and ele_dat is not None:
            print("WARNING: Glacier reference elevation is missing")
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
        set_up_end = (
            pd.to_datetime(set_up_start)
            + pd.DateOffset(years=1)
            + pd.DateOffset(days=-1)
        )
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
        raise ValueError(
            "WARNING: Resampling rate "
            + freq
            + " is not supported. Choose either 'D' (daily), 'W' (weekly), 'M' (monthly) or 'Y' (yearly)."
        )

    # Load parameter JSON
    parameters_path = importlib.resources.files("matilda") / "parameters.json"
    with parameters_path.open("r", encoding="utf-8") as file:
        parameter_data = json.load(file)["parameters"]

    # Build the parameter dictionary using defaults and passed matilda_param
    parameters = {}
    for param, properties in parameter_data.items():
        # Use explicitly passed value (matilda_param) or default from the JSON
        parameters[param] = matilda_param.get(param, properties["default"])

    # Validate parameters against bounds
    for param, value in parameters.items():
        bounds = parameter_data[param]
        min_value = bounds.get("min", float("-inf"))
        max_value = bounds.get("max", float("inf"))

        if not min_value <= value <= max_value:
            print(
                f"WARNING: Parameter {param} with value {value} exceeds "
                f"boundaries [{min_value}, {max_value}]. Using provided value."
            )

    # Compute derived parameters
    parameters["TT_rain"] = parameters.get("TT_snow") + parameters.get("TT_diff")
    parameters["CFMAX_ice"] = parameters.get("CFMAX_snow") * parameters.get("CFMAX_rel")

    # Create parameter series to pass to subsequent functions
    misc_param = {
        "set_up_start": set_up_start,
        "set_up_end": set_up_end,
        "sim_start": sim_start,
        "sim_end": sim_end,
        "freq": freq,
        "freq_long": freq_long,
        "lat": lat,
        "area_cat": area_cat,
        "area_glac": area_glac,
        "ele_dat": ele_dat,
        "ele_cat": ele_cat,
        "ele_glac": ele_glac,
        "ele_non_glac": ele_non_glac,
        "warn": warn,
        "CFR_ice": 0.01,  # fraction of ice melt refreezing in moulins
    }

    all_param = pd.Series({**misc_param, **parameters})

    print("Parameter set:")
    print(str(all_param))

    return all_param


def matilda_preproc(input_df, parameter, obs=None):
    """
    Processes and prepares input climate data and optional observation data for MATILDA simulations.
    Includes format transformations, unit conversions, and application of precipitation correction factors.

    Parameters
    ----------
    input_df : pandas.DataFrame or xarray.Dataset
        Input dataset containing climate variables such as temperature (`T2`) in Celsius or Kelvin
        and precipitation (`RRR`) in mm.
    parameter : pandas.Series
        Series of MATILDA parameters, including setup and simulation periods, precipitation correction factor,
        and additional configuration details.
    obs : pandas.DataFrame or None, optional
        Observation data for discharge, with columns 'Date' and 'Qobs' in m³/s. If provided, it will be
        resampled, converted to mm/day, and optionally filtered by season of interest (if specified). Defaults to None.

    Returns
    -------
    pandas.DataFrame
        Preprocessed input dataset with climate variables adjusted for the specified simulation period,
        including temperature conversion and precipitation correction.
    tuple of pandas.DataFrame, optional
        If `obs` is provided, returns a tuple of the preprocessed input dataset and the preprocessed
        observation dataset, with discharge values converted to mm/day and resampled to the simulation period.
    """

    print("*-------------------*")
    print("Reading data")
    print(
        "Set up period from "
        + str(parameter.set_up_start)
        + " to "
        + str(parameter.set_up_end)
        + " to set initial values"
    )
    print(
        "Simulation period from "
        + str(parameter.sim_start)
        + " to "
        + str(parameter.sim_end)
    )
    df_preproc = input_df.copy()

    if parameter.set_up_start > parameter.sim_start:
        print("WARNING: Spin up period starts after simulation period")
    elif isinstance(df_preproc, xr.Dataset):
        df_preproc = input_df.sel(time=slice(parameter.set_up_start, parameter.sim_end))
    else:
        df_preproc.set_index("TIMESTAMP", inplace=True)
        df_preproc.index = pd.to_datetime(df_preproc.index)
        df_preproc = df_preproc[parameter.set_up_start : parameter.sim_end]

    # make sure temperatures are in Celsius
    df_preproc["T2"] = np.where(
        df_preproc["T2"] >= 100, df_preproc["T2"] - 273.15, df_preproc["T2"]
    )

    # overall precipitation correction factor
    df_preproc["RRR"] = df_preproc["RRR"] * parameter.PCORR
    df_preproc["RRR"] = np.where(df_preproc["RRR"] < 0, 0, df_preproc["RRR"])

    if obs is not None:
        obs_preproc = obs.copy()
        obs_preproc.set_index("Date", inplace=True)
        obs_preproc.index = pd.to_datetime(obs_preproc.index)
        obs_preproc = obs_preproc[parameter.sim_start : parameter.sim_end]
        # Changing the input unit from m^3/s to mm/d.
        obs_preproc["Qobs"] = (
            obs_preproc["Qobs"] * 86400 / (parameter.area_cat * 1000000) * 1000
        )
        obs_preproc = obs_preproc.resample("D").agg(pd.Series.sum, skipna=False)
        # Expanding the observation period to full years filling up with NAs
        idx_first = obs_preproc.index.year[1]
        idx_last = obs_preproc.index.year[-1]
        idx = pd.date_range(
            start=date(idx_first, 1, 1),
            end=date(idx_last, 12, 31),
            freq="D",
            name=obs_preproc.index.name,
        )
        obs_preproc = obs_preproc.reindex(idx)
        obs_preproc = obs_preproc.fillna(np.NaN)

        print("Input data preprocessing successful")
        return df_preproc, obs_preproc

    print("Input data preprocessing successful")
    return df_preproc


def phase_separation(df_preproc, parameter):
    """
    Separates precipitation into liquid and solid fractions based on temperature thresholds using a linear transition.

    Parameters
    ----------
    df_preproc : pandas.DataFrame
        Preprocessed input dataset containing temperature (`T2`) in degrees Celsius and precipitation (`RRR`) in mm.
    parameter : pandas.Series
        Series of MATILDA parameters, including `TT_snow` (threshold temperature for snow),
        `TT_rain` (threshold temperature for rain), and other configuration details.

    Returns
    -------
    tuple of pandas.Series
        - `rain`: Liquid precipitation fraction (in mm).
        - `snow`: Solid precipitation fraction (in mm).
    """

    reduced_temp = (parameter.TT_rain - df_preproc["T2"]) / (
        parameter.TT_rain - parameter.TT_snow
    )
    snowfrac = np.clip(reduced_temp, 0, 1)
    snow = snowfrac * df_preproc["RRR"]
    rain = df_preproc["RRR"] - snow

    return rain, snow


def input_scaling(df_preproc, parameter):
    """
    Scales input climate data to mean elevations for glacierized and non-glacierized areas, separates precipitation
    into phases (rain and snow), and applies the snowfall correction factor.

    Parameters
    ----------
    df_preproc : pandas.DataFrame
        Preprocessed input dataset containing temperature (`T2`) in degrees Celsius and precipitation (`RRR`) in mm.
    parameter : pandas.Series
        Series of MATILDA parameters, including:
            - `ele_glac`: Mean elevation of the glacierized area (in meters).
            - `ele_non_glac`: Mean elevation of the non-glacierized area (in meters).
            - `ele_dat`: Reference elevation of the input data (in meters).
            - `lr_temp`: Temperature lapse rate (°C/m).
            - `lr_prec`: Precipitation lapse rate (mm/m).
            - `pfilter`: Threshold for filtering small precipitation values (mm).
            - `SFCF`: Snowfall correction factor.

    Returns
    -------
    tuple of pandas.DataFrame
        - `input_df_glacier`: Scaled dataset for the glacierized area, including separated rain and snow fractions.
        - `input_df_catchment`: Scaled dataset for the non-glacierized catchment, including separated rain and snow fractions.
    """

    if parameter.ele_glac is not None:
        elev_diff_glacier = parameter.ele_glac - parameter.ele_dat
        input_df_glacier = df_preproc.copy()
        input_df_glacier["T2"] = input_df_glacier["T2"] + elev_diff_glacier * float(
            parameter.lr_temp
        )
        input_df_glacier["RRR"] = np.where(
            input_df_glacier["RRR"] > parameter.pfilter,
            # Apply precipitation lapse rate only, when there is precipitation!
            input_df_glacier["RRR"] + elev_diff_glacier * float(parameter.lr_prec),
            0,
        )
        input_df_glacier["RRR"] = np.where(
            input_df_glacier["RRR"] < 0, 0, input_df_glacier["RRR"]
        )
    else:
        input_df_glacier = df_preproc.copy()
    if parameter.ele_non_glac is not None:
        elev_diff_catchment = parameter.ele_non_glac - parameter.ele_dat
        input_df_catchment = df_preproc.copy()
        input_df_catchment["T2"] = input_df_catchment[
            "T2"
        ] + elev_diff_catchment * float(parameter.lr_temp)
        input_df_catchment["RRR"] = np.where(
            input_df_catchment["RRR"] > parameter.pfilter,
            # Apply precipitation lapse rate only, when there is precipitation!
            input_df_catchment["RRR"] + elev_diff_catchment * float(parameter.lr_prec),
            0,
        )
        input_df_catchment["RRR"] = np.where(
            input_df_catchment["RRR"] < 0, 0, input_df_catchment["RRR"]
        )

    else:
        input_df_catchment = df_preproc.copy()

    # precipitation phase separation:
    input_df_glacier["rain"], input_df_glacier["snow"] = phase_separation(
        input_df_glacier, parameter
    )
    input_df_catchment["rain"], input_df_catchment["snow"] = phase_separation(
        input_df_catchment, parameter
    )

    # snow correction factor
    input_df_glacier["snow"], input_df_catchment["snow"] = [
        parameter.SFCF * i
        for i in [input_df_glacier["snow"], input_df_catchment["snow"]]
    ]

    # add corrected snow fall to total precipitation
    input_df_glacier["RRR"], input_df_catchment["RRR"] = [
        i["snow"] + i["rain"] for i in [input_df_glacier, input_df_catchment]
    ]

    return input_df_glacier, input_df_catchment


def calculate_PDD(ds, prints=True):
    """
    Calculates positive degree days (PDD) from a provided time series dataset, along with daily means of temperature and
    precipitation components (rain and snow).

    Parameters
    ----------
    ds : xarray.Dataset or pandas.DataFrame
        Input dataset containing temperature (`T2`), total precipitation (`RRR`), and optionally rain and snow variables.
        For xarray datasets, a glacier mask (`MASK`) is applied if present.
    parameter : pandas.Series
        Series of MATILDA parameters (not directly used in this function but required for compatibility).
    prints : bool, optional
        If True, prints status messages during the calculation. Defaults to True.

    Returns
    -------
    xarray.Dataset
        A dataset containing:
            - `temp_mean`: Daily mean temperature in °C.
            - `RRR`: Daily total precipitation in mm.
            - `rain`: Daily rainfall in mm.
            - `snow`: Daily snowfall in mm.
            - `pdd`: Positive degree days (sum of daily mean temperatures above 0°C).
    """

    if prints:
        print("*-------------------*")
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

    pdd_ds = xr.merge(
        [
            xr.DataArray(temp_mean, name="temp_mean"),
            xr.DataArray(prec),
            xr.DataArray(rain),
            xr.DataArray(snow),
        ]
    )

    # calculate the positive degree days
    pdd_ds["pdd"] = xr.where(pdd_ds["temp_mean"] > 0, pdd_ds["temp_mean"], 0)

    return pdd_ds


def melt_rates(snow, pdd, parameter):
    """
    Computes melt rates from snow precipitation and positive degree day (PDD) sums.
    Snow melt is calculated using the `CFMAX_snow` parameter, and excess energy after snow melt contributes to ice melt,
    calculated using the `CFMAX_ice` parameter.

    This function is adapted from `pypdd.py`, line 331. Original source is referenced at the beginning of the script.

    Parameters
    ----------
    snow : array_like
        Snow precipitation rate (mm/day).
    pdd : array_like
        Positive degree day values (°C·days).
    parameter : pandas.Series
        Series of MATILDA parameters containing:
            - `CFMAX_snow`: Degree-day factor for snowmelt.
            - `CFMAX_ice`: Degree-day factor for ice melt.

    Returns
    -------
    tuple of array_like
        - `snow_melt`: Effective snow melt rates (mm/day), limited by the available snow.
        - `ice_melt`: Ice melt rates (mm/day), proportional to excess PDD energy beyond snow melt.
    """

    # compute a potential snow melt
    pot_snow_melt = parameter.CFMAX_snow * pdd
    # effective snow melt can't exceed amount of snow
    snow_melt = np.minimum(snow, pot_snow_melt)
    # ice melt is proportional to excess snow melt
    ice_melt = (pot_snow_melt - snow_melt) * parameter.CFMAX_ice / parameter.CFMAX_snow
    # return melt rates
    return (snow_melt, ice_melt)


def calculate_glaciermelt(ds, parameter, prints=True):
    """
    Calculates accumulation, snow and ice melt, and runoff rates from glaciers using a Degree Day Model (DDM).
    This method is inspired by PYPDD (github.com/juseg/pypdd) and includes glacier storage-release dynamics.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset containing daily climate variables:
            - `temp_mean`: Daily mean temperature in °C.
            - `RRR`: Total precipitation in mm.
            - `snow`: Daily snowfall in mm.
            - `rain`: Daily rainfall in mm.
            - `pdd`: Positive degree day values (°C·days).
    parameter : pandas.Series
        Series of MATILDA parameters containing:
            - `CFMAX_snow`: Degree-day factor for snowmelt.
            - `CFMAX_ice`: Degree-day factor for ice melt.
            - `CFR`: Refreezing factor for snowmelt.
            - `CFR_ice`: Refreezing factor for ice melt.
            - `AG`: Glacier outflow adjustment parameter.
    prints : bool, optional
        If True, prints status updates during the calculation. Defaults to True.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing daily values for various outputs of the Degree Day Model:
            - `DDM_smb`: Surface mass balance (mm w.e.).
            - `pdd`: Positive degree days (°C·days).
            - `DDM_temp`: Daily mean temperature (°C).
            - `DDM_prec`: Total precipitation (mm).
            - `DDM_rain`: Rainfall (mm).
            - `DDM_snow`: Snowfall (mm).
            - `DDM_accumulation_rate`: Snow accumulation rate (mm).
            - `DDM_ice_melt`: Ice melt (mm).
            - `DDM_snow_melt`: Snow melt (mm).
            - `DDM_total_melt`: Total melt (mm).
            - `DDM_refreezing`: Total refreezing (mm).
            - `DDM_glacier_reservoir`: Water stored in the glacier reservoir (mm).
            - `Q_DDM`: Actual runoff from the glacier (mm).

    References
    ----------
    - PYPDD (github.com/juseg/pypdd)
    - Stahl, K., Moore, R. D., & McKendry, I. G. (2008). The role of synoptic-scale circulation in the linkage between large-scale ocean-atmosphere indices and winter surface climate in British Columbia, Canada. *Water Resources Research*, 44(7). https://doi.org/10.1029/2007WR005956
    - Toum, J., et al. (2021). Understanding glacier mass-balance variability using climate teleconnections. *The R Journal*, 13(1). https://doi.org/10.32614/RJ-2021-059

    Notes
    -----
    This implementation uses a storage-release scheme for glacier outflow that adjusts based on snow depth
    and drainage system conditions.
    """

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
    snow_depth = np.array(snow_depth)  # not actual depth but mm w.e.!
    ice_melt = np.array(ice_melt)
    snow_melt = np.array(snow_melt)

    # calculate refreezing, runoff and surface mass balance
    total_melt = snow_melt + ice_melt
    refr_ice = parameter.CFR_ice * ice_melt
    refr_snow = parameter.CFR * snow_melt
    refr_tot = refr_snow + refr_ice
    runoff_rate = total_melt - refr_tot
    inst_smb = accu_rate - runoff_rate
    runoff_rate_rain = runoff_rate + rain

    # Storage-release scheme for glacier outflow (Stahl et.al. 2008, Toum et. al. 2021)
    KG_min = 0.1  # minimum outflow coefficient (conditions with deep snow and poorly developed glacial drainage systems) [time^−1]
    d_KG = 0.9  # KG_min + d_KG = maximum outflow coefficient (representing late-summer conditions with bare ice and a well developed glacial drainage system) [time^−1]
    KG = np.minimum(
        KG_min + d_KG * np.exp(snow_depth / -(0.1 * 1000000**parameter.AG)), 1
    )
    for i in np.arange(len(temp)):
        if i == 0:
            SG = runoff_rate_rain[i]  # liquid water stored in the reservoir
        else:
            SG = np.maximum((runoff_rate_rain[i] - actual_runoff[i - 1]) + SG, 0)
        actual_runoff.append(KG[i] * SG)
        glacier_reservoir.append(SG)

    # final glacier module output (everything but temperature and pdd in mm w.e.)
    glacier_melt = xr.merge(
        [
            xr.DataArray(inst_smb, name="DDM_smb"),
            xr.DataArray(pdd, name="pdd"),
            xr.DataArray(temp, name="DDM_temp"),
            xr.DataArray(prec, name="DDM_prec"),
            xr.DataArray(rain, name="DDM_rain"),
            xr.DataArray(snow, name="DDM_snow"),
            xr.DataArray(accu_rate, name="DDM_accumulation_rate"),
            xr.DataArray(ice_melt, name="DDM_ice_melt"),
            xr.DataArray(snow_melt, name="DDM_snow_melt"),
            xr.DataArray(total_melt, name="DDM_total_melt"),
            xr.DataArray(refr_tot, name="DDM_refreezing"),
            xr.DataArray(glacier_reservoir, name="DDM_glacier_reservoir"),
            xr.DataArray(actual_runoff, name="Q_DDM"),
        ]
    )

    DDM_results = glacier_melt.to_dataframe()

    # merged data array comes without index -> set DateTime index from input
    idx = ds.coords.to_index()
    DDM_results = DDM_results.set_index(pd.DatetimeIndex(idx))

    if prints:
        print("Finished Degree-Day Melt Routine")

    return DDM_results


def create_lookup_table(glacier_profile, parameter):
    """
    Generates a lookup table of glacier area and water equivalent changes from the initial state (100% glacier coverage)
    to an ice-free state (0% coverage) using the deltaH scaling approach. This method is based on the routines outlined
    in Seibert et al. (2018) and Huss et al. (2010).

    Parameters
    ----------
    glacier_profile : pandas.DataFrame
        DataFrame containing the glacier's initial state, including:
            - `Area`: Area of each elevation band (in km²).
            - `WE`: Initial water equivalent of each elevation band (in mm w.e.).
            - `Elevation`: Elevation of each band (in meters).
    parameter : pandas.Series
        Series of MATILDA parameters, including:
            - `area_glac`: Total glacier area (in km²).

    Returns
    -------
    pandas.DataFrame
        Lookup table showing scaled glacier area for each elevation band over 101 mass states (from 100% to 0% in 1% steps).
        Each column corresponds to an elevation band (`EleZone`), and each row represents a scaled mass state.

    References
    ----------
    - Huss, M., Jouvet, G., Farinotti, D., & Bauder, A. (2010). Future high-mountain hydrology: a new parameterization
      of glacier retreat. *Hydrology and Earth System Sciences, 14*(5), 815–829.
      https://doi.org/10.5194/hess-14-815-2010
    - Seibert, J., Vis, M. J. P., Kohn, I., Weiler, M., & Stahl, K. (2018). Technical note: Representing glacier geometry
      changes in a semi-distributed hydrological model. *Hydrology and Earth System Sciences, 22*(4), 2211–2224.
      https://doi.org/10.5194/hess-22-2211-2018

    Notes
    -----
    1. The deltaH parameterization involves a scaling factor (`fs`) based on the total glacier mass change and the
       normalized elevation profile of the glacier.
    2. Three different parameter sets (`a`, `b`, `c`, `y`) are applied based on glacier size as outlined in Huss et al. (2010).
    3. Elevation bands with negative water equivalent are excluded iteratively during the scaling process.
    """

    initial_area = glacier_profile["Area"]  # per elevation band
    hi_initial = glacier_profile[
        "WE"
    ]  # initial water equivalent of each elevation band
    hi_k = glacier_profile[
        "WE"
    ]  # hi_k is the updated water equivalent for each elevation zone, starts with initial values
    ai = glacier_profile[
        "Area"
    ]  # ai is the glacier area of each elevation zone, starts with initial values

    lookup_table = pd.DataFrame()
    lookup_table = pd.concat(
        [lookup_table, initial_area.to_frame().T], ignore_index=True
    )

    # Pre-simulation
    # 1. calculate total glacier mass in mm water equivalent: M = sum(ai * hi)
    m = sum(glacier_profile["Area"] * glacier_profile["WE"])

    # melt the glacier in steps of 1 percent
    deltaM = -m / 100

    # 2. Normalize glacier elevations: Einorm = (Emax-Ei)/(Emax-Emin)
    glacier_profile["norm_elevation"] = (
        glacier_profile["Elevation"].max() - glacier_profile["Elevation"]
    ) / (glacier_profile["Elevation"].max() - glacier_profile["Elevation"].min())
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

    glacier_profile["delta_h"] = (
        (glacier_profile["norm_elevation"] + a) ** y
        + (b * (glacier_profile["norm_elevation"] + a))
        + c
    )

    ai_scaled = ai.copy()  # initial values set as ai_scaled

    fs = deltaM / (sum(ai * glacier_profile["delta_h"]))  # a) initial ai

    for _ in range(99):
        # 5. compute glacier geometry for reduced mass
        hi_k = hi_k + fs * glacier_profile["delta_h"]

        leftover = sum(
            pd.Series(np.where(hi_k < 0, hi_k, 0)) * ai
        )  # Calculate leftover (i.e. the 'negative' glacier volume)

        hi_k = pd.Series(
            np.where(hi_k < 0, np.nan, hi_k)
        )  # Set those zones that have a negative we to NaN to make sure they will be excluded from now on

        # 6. width scaling
        ai_scaled = ai * np.sqrt((hi_k / hi_initial))

        # 7. create lookup table
        # glacier area for each elevation band for 101 different mass situations (100 percent to 0 in 1 percent steps)

        lookup_table = pd.concat(
            [lookup_table, ai_scaled.to_frame().T], ignore_index=True
        )

        if (
            sum(
                pd.Series(np.where(np.isnan(ai_scaled), 0, ai))
                * glacier_profile["delta_h"]
            )
            == 0
        ):
            ai_scaled = np.where(ai_scaled == 1, 1, 0)
        else:
            # Update fs (taking into account the leftover)
            fs = (deltaM + leftover) / sum(
                pd.Series(np.where(np.isnan(ai_scaled), 0, ai))
                * glacier_profile["delta_h"]
            )

    lookup_table = lookup_table.fillna(0)

    lookup_table.columns = glacier_profile["EleZone"]
    lookup_table = lookup_table.groupby(level=0, axis=1).sum()

    elezones_inital = lookup_table.iloc[0]

    lookup_table = lookup_table / elezones_inital
    lookup_table = round(lookup_table, 4)
    lookup_table.iloc[-1] = 0
    return lookup_table


def glacier_area_change(output_DDM, lookup_table, glacier_profile, parameter):
    """
    Calculates the new glacier area for each hydrological year and scales glacier variables using the deltaH scaling
    approach. This is the second part of the glacier scaling routine, based on Seibert et al. (2018) and Huss et al. (2010).

    Parameters
    ----------
    output_DDM : pandas.DataFrame
        Degree Day Model (DDM) output containing surface mass balance (`DDM_smb`) and other glacier-related variables.
    lookup_table : pandas.DataFrame
        Lookup table created in part 1 of the scaling routine, mapping glacier area changes to mass loss percentages.
    glacier_profile : pandas.DataFrame
        DataFrame containing glacier initial states:
            - `Area`: Area of each elevation band (in km²).
            - `WE`: Initial water equivalent of each elevation band (in mm w.e.).
    parameter : pandas.Series
        Series of MATILDA parameters, including:
            - `area_glac`: Total glacier area (in km²).
            - `area_cat`: Catchment area (in km²).
            - `hydro_year`: Starting month of the hydrological year (1–12).

    Returns
    -------
    tuple
        - `output_DDM` (pandas.DataFrame): Updated DDM output with scaled glacier variables for each hydrological year.
        - `glacier_change_area` (pandas.DataFrame): Time series of annual glacier area changes and cumulative scaled SMB.

    References
    ----------
    - Huss, M., Jouvet, G., Farinotti, D., & Bauder, A. (2010). Future high-mountain hydrology: a new parameterization
      of glacier retreat. *Hydrology and Earth System Sciences, 14*(5), 815–829.
      https://doi.org/10.5194/hess-14-815-2010
    - Seibert, J., Vis, M. J. P., Kohn, I., Weiler, M., & Stahl, K. (2018). Technical note: Representing glacier geometry
      changes in a semi-distributed hydrological model. *Hydrology and Earth System Sciences, 22*(4), 2211–2224.
      https://doi.org/10.5194/hess-22-2211-2018

    Notes
    -----
    1. The glacier area change is calculated annually based on the cumulative surface mass balance (SMB).
    2. SMB is scaled to the current glacier area to reflect its dynamic changes.
    3. If the cumulative SMB indicates a mass loss beyond 99% of the initial mass, the glacier area is set to zero.
    4. Variables in `output_DDM` are updated to reflect the new glacierized area fraction each year.
    """

    # select output columns to update
    up_cols = output_DDM.columns.drop(["DDM_smb", "DDM_temp", "pdd"])

    # creating columns for updated DDM output
    for col in up_cols:
        output_DDM[col + "_updated_scaled"] = copy.deepcopy(output_DDM[col])

    # determining the hydrological year
    output_DDM["water_year"] = np.where(
        (output_DDM.index.month) >= parameter.hydro_year,
        output_DDM.index.year + 1,
        output_DDM.index.year,
    )

    # initial glacier mass from the glacier profile in mm w.e. (relative to the whole catchment)
    m = sum((glacier_profile["Area"]) * glacier_profile["WE"])

    # initial area
    initial_area = glacier_profile.groupby("EleZone")["Area"].sum()

    # dataframe with the smb change per hydrological year in mm w.e.
    glacier_change = pd.DataFrame(
        {"smb": output_DDM.groupby("water_year")["DDM_smb"].sum()}
    ).reset_index()

    initial_year = output_DDM["water_year"].iloc[0]
    glacier_change_area = pd.DataFrame(
        {"time": initial_year, "glacier_area": [parameter.area_glac]}
    )

    # setting initial values for the loop
    new_area = parameter.area_glac
    smb_cum = 0
    i = 1
    for i in range(len(glacier_change)):
        year = glacier_change["water_year"][i]
        smb = glacier_change["smb"][i]
        # scale the smb to the (updated) glacierized fraction of the catchment
        smb = smb * (
            new_area / parameter.area_cat
        )  # SMB is area (re-)scaled because m is area scaled as well
        # add the smb from the previous year(s) to the new year
        smb_cum = smb_cum + smb
        # calculate the percentage of melt in comparison to the initial mass
        smb_percentage = round((-smb_cum / m) * 100)
        if (smb_percentage <= 99) & (smb_percentage >= 0):
            # select the correct row from the lookup table depending on the smb
            area_melt = lookup_table.iloc[smb_percentage]
            # derive the new glacier area by multiplying the initial area with the area changes
            new_area = (
                np.nansum((area_melt.values * initial_area.values)) * parameter.area_cat
            )
        else:
            new_area = 0
        # scale the output with the new glacierized area
        glacier_change_area = glacier_change_area.append(
            {"time": year, "glacier_area": new_area, "smb_scaled_cum": smb_cum},
            ignore_index=True,
        )
        for col in up_cols:
            output_DDM[col + "_updated_scaled"] = np.where(
                output_DDM["water_year"] == year,
                output_DDM[col] * (new_area / parameter.area_cat),
                output_DDM[col + "_updated_scaled"],
            )

        glacier_change_area["time"] = pd.to_datetime(
            glacier_change_area["time"], format="%Y"
        )
        glacier_change_area.set_index("time", inplace=True, drop=False)
        glacier_change_area["time"] = glacier_change_area["time"].dt.strftime("%Y")
        glacier_change = glacier_change.rename_axis("TIMESTAMP")

    return output_DDM, glacier_change_area


def updated_glacier_melt(
    data, lookup_table, glacier_profile, parameter, drop_surplus=False
):
    """
    Calculates the new glacier area for each hydrological year and scales glacier variables using the deltaH scaling
    approach. This is the second part of the glacier scaling routine, based on Seibert et al. (2018) and Huss et al. (2010).

    Parameters
    ----------
    output_DDM : pandas.DataFrame
        Degree Day Model (DDM) output containing surface mass balance (`DDM_smb`) and other glacier-related variables.
    lookup_table : pandas.DataFrame
        Lookup table created in part 1 of the scaling routine, mapping glacier area changes to mass loss percentages.
    glacier_profile : pandas.DataFrame
        DataFrame containing glacier initial states:
            - `Area`: Area of each elevation band (in km²).
            - `WE`: Initial water equivalent of each elevation band (in mm w.e.).
    parameter : pandas.Series
        Series of MATILDA parameters, including:
            - `area_glac`: Total glacier area (in km²).
            - `area_cat`: Catchment area (in km²).
            - `hydro_year`: Starting month of the hydrological year (1–12).

    Returns
    -------
    tuple
        - `output_DDM` (pandas.DataFrame): Updated DDM output with scaled glacier variables for each hydrological year.
        - `glacier_change_area` (pandas.DataFrame): Time series of annual glacier area changes and cumulative scaled SMB.

    References
    ----------
    - Huss, M., Jouvet, G., Farinotti, D., & Bauder, A. (2010). Future high-mountain hydrology: a new parameterization
      of glacier retreat. *Hydrology and Earth System Sciences, 14*(5), 815–829.
      https://doi.org/10.5194/hess-14-815-2010
    - Seibert, J., Vis, M. J. P., Kohn, I., Weiler, M., & Stahl, K. (2018). Technical note: Representing glacier geometry
      changes in a semi-distributed hydrological model. *Hydrology and Earth System Sciences, 22*(4), 2211–2224.
      https://doi.org/10.5194/hess-22-2211-2018

    Notes
    -----
    1. The glacier area change is calculated annually based on the cumulative surface mass balance (SMB).
    2. SMB is scaled to the current glacier area to reflect its dynamic changes.
    3. If the cumulative SMB indicates a mass loss beyond 99% of the initial mass, the glacier area is set to zero.
    4. Variables in `output_DDM` are updated to reflect the new glacierized area fraction each year.
    """

    # determine hydrological years
    data["water_year"] = np.where(
        (data.index.month) >= parameter.hydro_year, data.index.year + 1, data.index.year
    )

    # initial glacier mass from the glacier profile in mm w.e. (relative to the whole catchment)
    m = np.nansum((glacier_profile["Area"]) * glacier_profile["WE"])

    # initial area
    initial_area = glacier_profile.groupby("EleZone")["Area"].sum()

    # re-calculate the mean glacier elevation based on the glacier profile in rough elevation zones for consistency (method outlined in following loop)
    print("Recalculating initial elevations based on glacier profile")
    init_dist = (
        initial_area.values / initial_area.values.sum()
    )  # fractions of glacierized area elev zones
    init_elev = (
        init_dist * lookup_table.columns.values
    )  # multiply fractions with average zone elevations
    init_elev = int(init_elev.sum())
    print(">> Prior glacier elevation: " + str(parameter.ele_glac) + "m a.s.l.")
    print(">> Recalculated glacier elevation: " + str(init_elev) + "m a.s.l.")

    # re-calculate the mean non-glacierized elevation accordingly
    if parameter.ele_cat is None:
        ele_non_glac = None
    else:
        ele_non_glac = (
            (parameter.ele_cat - parameter.area_glac / parameter.area_cat * init_elev)
            * parameter.area_cat
            / (parameter.area_cat - parameter.area_glac)
        )
    if ele_non_glac is not None:
        print(
            ">> Prior non-glacierized elevation: "
            + str(round(parameter.ele_non_glac))
            + "m a.s.l."
        )
        print(
            ">> Recalculated non-glacierized elevation: "
            + str(round(ele_non_glac))
            + "m a.s.l."
        )

    # Setup initial variables for main loop
    new_area = parameter.area_glac
    smb_cum = 0
    surplus = 0
    warn = True
    output_DDM = pd.DataFrame()
    parameter_updated = copy.deepcopy(parameter)
    parameter_updated.ele_glac = init_elev
    parameter_updated.ele_non_glac = ele_non_glac

    # create initial non-updated dataframes
    input_df_glacier, input_df_catchment = input_scaling(data, parameter_updated)

    # Slice input data to simulation period (with full hydrological years if the setup period allows it)
    if datetime.fromisoformat(parameter.sim_start).month < parameter.hydro_year:
        startyear = data[parameter.sim_start : parameter.sim_end].water_year[0] - 1
    else:
        startyear = data[parameter.sim_start : parameter.sim_end].water_year[0]

    startdate = str(startyear) + "-" + str(parameter.hydro_year) + "-" + "01"

    if datetime.fromisoformat(startdate) < datetime.fromisoformat(
        parameter.set_up_start
    ):
        # Provided setup period does not cover the full hydrological year sim_start is in
        data_update = data[parameter.sim_start : parameter.sim_end]
        input_df_glacier = input_df_glacier[parameter.sim_start : parameter.sim_end]
        input_df_catchment_spinup = input_df_catchment[
            parameter.set_up_start : parameter.sim_start
        ]
        input_df_catchment = input_df_catchment[parameter.sim_start : parameter.sim_end]

        print(
            "**********\n"
            "WARNING!\n"
            "The provided setup period does not cover the full hydrological year the simulation period \n"
            "starts in. The initial surface mass balance (SMB) of the first hydrological year in the glacier \n"
            "rescaling routine therefore possibly misses a significant part of the accumulation period (e.g. Oct-Dec).\n"
            "**********\n"
        )
    else:
        data_update = data[startdate : parameter.sim_end]
        input_df_glacier = input_df_glacier[startdate : parameter.sim_end]
        input_df_catchment_spinup = input_df_catchment[
            parameter.set_up_start : startdate
        ]
        input_df_catchment = input_df_catchment[startdate : parameter.sim_end]

    # create initial df of glacier change
    glacier_change = pd.DataFrame(
        {
            "time": startyear,
            "glacier_area": [parameter.area_glac],
            "glacier_elev": init_elev,
        }
    )

    # Loop through simulation period annually updating catchment fractions and scaling elevations
    if parameter.ele_dat is None:
        raise ValueError(
            "You need to provide ele_dat in order to apply the glacier-rescaling routine."
        )
    print("Calculating glacier evolution")
    for i in range(len(data_update.water_year.unique())):
        year = data_update.water_year.unique()[i]
        mask = data_update.water_year == year

        # Use updated glacier area of the previous year
        parameter_updated.area_glac = new_area
        # Use updated glacier elevation of the previous year
        if i != 0:
            parameter_updated.ele_glac = new_distribution

        # Calculate the updated mean elevation of the non-glacierized catchment area
        if parameter_updated.ele_cat is None:
            parameter_updated.ele_non_glac = None
        else:
            parameter_updated.ele_non_glac = (
                (
                    parameter_updated.ele_cat
                    - parameter_updated.area_glac
                    / parameter_updated.area_cat
                    * parameter_updated.ele_glac
                )
                * parameter_updated.area_cat
                / (parameter_updated.area_cat - parameter_updated.area_glac)
            )

        # Scale glacier and hbv routine inputs in selected year with updated parameters
        input_df_glacier[mask], input_df_catchment[mask] = input_scaling(
            data_update[mask], parameter_updated
        )

        # Calculate positive degree days and glacier ablation/accumulation
        degreedays_ds = calculate_PDD(input_df_glacier[mask], prints=False)
        output_DDM_year = calculate_glaciermelt(
            degreedays_ds, parameter_updated, prints=False
        )
        output_DDM_year["water_year"] = data_update.water_year[mask]

        # deselect output columns not to update
        up_cols = output_DDM_year.columns.drop(
            ["DDM_smb", "DDM_temp", "pdd", "water_year"]
        )

        # create columns for updated DDM output
        for col in up_cols:
            output_DDM_year[col + "_updated_scaled"] = copy.deepcopy(
                output_DDM_year[col]
            )

        # Rescale glacier geometry and update glacier parameters in all but the last (incomplete) water year
        if i < len(data_update.water_year.unique()) - 1:

            smb_unscaled = output_DDM_year["DDM_smb"].sum()

            # if True: model runs with positive MB_cum at any time are 'dropped' (runoff = 0.01, SMB 9999)
            if drop_surplus:

                if i == 0 and smb_unscaled > 0:
                    print(
                        "**********\n"
                        "WARNING:\n"
                        "The cumulative surface mass balance in the first year of the simulation period is \n"
                        "positive. You may want to shift the starting year or set drop_surplus=False.\n"
                        "**********\n"
                    )
                # scale the smb to the (updated) glacierized fraction of the catchment
                smb = smb_unscaled * (
                    new_area / parameter.area_cat
                )  # SMB is area (re-)scaled because m is area scaled as well
                # add the smb from the previous year(s) to the new year
                smb_cum = smb_cum + smb
                if smb_cum > 0:
                    if warn:
                        print(
                            "**********\n"
                            "WARNING:\n"
                            "The cumulative surface mass balance in the simulation period is positive. \n"
                            "The glacier rescaling routine cannot model glacier extent exceeding the initial status of \n"
                            "the provided glacier profile. In order to exclude this run from parameter optimization \n"
                            "routines, a flag is passed, simulated runoff is set to 0.01, and SMB to 9999. \n"
                            "If you want to maintain the average mass balance set drop_surplus=False.\n"
                            "**********\n"
                        )
                        warn = False
                    smb_cum = m
                    new_distribution = parameter.ele_glac
                    smb_flag = True
                else:
                    smb_flag = False

            # if drop_surplus = False the surplus from years with positive MB_cum is added in later years
            else:
                # scale the smb to the (updated) glacierized fraction of the catchment
                smb = smb_unscaled * (
                    new_area / parameter.area_cat
                )  # SMB is area (re-)scaled because m is area scaled as well
                smb_scaled = smb.copy()

                # If the cumulative SMB has been positive in previous years the surplus is added here
                if surplus > 0:
                    if smb < 0:
                        diff = surplus + smb
                        surplus = max(diff, 0)
                        smb = min(diff, 0)
                # add the smb from the previous year(s) to the new year
                smb_cum = smb_cum + smb
                # Check whether glacier extent exceeds the initial state (smb_cum > 0). Shift surplus to next year(s).
                if smb_cum > 0:
                    if warn:
                        print(
                            "**********\n"
                            "WARNING:\n"
                            "At some point of the simulation period the cumulative surface mass balance is\n"
                            " positive. The glacier rescaling routine cannot model glacier extent exceeding the initial\n"
                            " status of the provided glacier profile. The surplus is stored and added in subsequent years\n"
                            " with negative mass balance(s) to maintain the long-term average balance.\n"
                            "**********\n"
                        )
                        warn = False
                    surplus += max(smb_cum, 0)
                    smb_cum = 0

                smb_flag = False

            # calculate the percentage of melt in comparison to the initial mass
            smb_percentage = round((-smb_cum / m) * 100)
            if (smb_percentage < 99) & (smb_percentage >= 0):
                # select the correct row from the lookup table depending on the smb
                area_melt = lookup_table.iloc[smb_percentage]
                # derive the new glacier area by multiplying the initial area with the area changes
                new_area = (
                    np.nansum((area_melt.values * initial_area.values))
                    * parameter.area_cat
                )
                # derive new spatial distribution of glacierized area (relative fraction in every elevation zone)
                new_distribution = (
                    (area_melt.values * initial_area.values) * parameter.area_cat
                ) / new_area
                # multiply relative portions with mean zone elevations to get rough estimate for new mean elevation
                new_distribution = (
                    new_distribution * lookup_table.columns.values
                )  # column headers contain elevations
                new_distribution = int(np.nansum(new_distribution))
            else:
                new_area = 0

            glacier_mass_abs = (1 - smb_percentage * 0.01) * m
            glacier_vol_init = (
                (m / 1000) * parameter.area_glac * 1e6 / 0.908
            )  # mass in mmwe, area in km^2
            glacier_vol = (glacier_mass_abs / 1000) * new_area * 1e6 / 0.908
            glacier_vol_perc = glacier_vol / glacier_vol_init

            # Append to glacier change dataframe for subsequent functions (skip last incomplete year)
            data = {
                "time": year,
                "glacier_area": new_area,
                "glacier_elev": new_distribution,
                "smb_water_year": smb_unscaled,
            }

            if drop_surplus:
                data["smb_scaled_cum"] = smb_cum
            else:
                data.update(
                    {
                        "smb_scaled": smb_scaled,
                        "smb_scaled_capped": smb,
                        "smb_scaled_capped_cum": smb_cum,
                        "surplus": surplus,
                        "glacier_melt_perc": smb_percentage,
                        "glacier_mass_mmwe": glacier_mass_abs,
                        "glacier_vol_m3": glacier_vol,
                        "glacier_vol_perc": glacier_vol_perc,
                    }
                )

            # Create the DataFrame and concatenate
            new_row = pd.DataFrame(data, index=[i])
            glacier_change = pd.concat([glacier_change, new_row], ignore_index=True)

        # Scale DDM output to new glacierized fraction
        for col in up_cols:
            output_DDM_year[col + "_updated_scaled"] = np.where(
                output_DDM_year["water_year"] == year,
                output_DDM_year[col] * (new_area / parameter.area_cat),
                output_DDM_year[col + "_updated_scaled"],
            )
        # Append year to full dataset
        output_DDM = pd.concat([output_DDM, output_DDM_year])

        if smb_flag:
            output_DDM["smb_flag"] = 1
            output_DDM["DDM_smb"] = (
                9999  # To exclude run from parameter optimization of glacial parameters
            )

    glacier_change["time"] = pd.to_datetime(glacier_change["time"], format="%Y")
    glacier_change.set_index("time", inplace=True, drop=False)
    glacier_change["time"] = glacier_change["time"].dt.strftime("%Y")
    glacier_change = glacier_change.rename_axis("TIMESTAMP")

    output_DDM = output_DDM[parameter.sim_start : parameter.sim_end]
    # Add original spin-up period back to HBV input
    input_df_catchment = pd.concat([input_df_catchment_spinup, input_df_catchment])

    return output_DDM, glacier_change, input_df_catchment


def hbv_simulation(input_df_catchment, parameter, glacier_area=None):
    """
    Simulates runoff from a catchment using the HBV model. Calculates key hydrological processes, including snowmelt,
    evaporation, soil moisture, and groundwater flow, based on input climate data and parameters.

    The Python code is adapted from the LHMP hydrological modeling playground
    (github.com/hydrogo/LHMP, Ayzel Georgy. (2016). doi:10.5281/zenodo.59501).
    Evapotranspiration values are calculated using the method outlined in Oudin et al. (2005) if not provided in the input data.

    Parameters
    ----------
    input_df_catchment : pandas.DataFrame
        Input climate dataset with daily resolution, including:
            - `T2`: Temperature (°C).
            - `RRR`: Total precipitation (mm).
            - `rain`: Rainfall (mm).
            - `snow`: Snowfall (mm).
            - `PE` (optional): Potential evapotranspiration (mm).
    parameter : pandas.Series
        Series of HBV model parameters, including:
            - `CFMAX_snow`: Degree-day factor for snowmelt.
            - `CFR`: Refreezing factor for snowmelt.
            - `TT_snow`: Threshold temperature for snowmelt.
            - `CWH`: Water holding capacity of snowpack.
            - `FC`: Field capacity of the soil.
            - `LP`: Soil moisture threshold for potential evapotranspiration.
            - `BETA`: Shape parameter for soil moisture recharge.
            - `PERC`: Percolation rate from upper to lower groundwater box.
            - `K0`, `K1`, `K2`: Recession coefficients for runoff components.
            - `UZL`: Threshold for upper groundwater runoff.
            - `CET`: Correction factor for evapotranspiration.
            - `MAXBAS`: Parameter for hydrograph smoothing.
            - `area_cat`: Total catchment area (km²).
            - `area_glac`: Glacierized area within the catchment (km²).
            - `lat`: Latitude of the catchment for radiation calculations.
            - `sim_start`, `sim_end`: Simulation period (YYYY-MM-DD).
            - `set_up_start`, `set_up_end`: Setup period (YYYY-MM-DD).
    glacier_area : pandas.DataFrame, optional
        Time series of annual glacier areas for dynamically scaling snow and rain fractions. Defaults to None.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing simulated hydrological variables and runoff, including:
            - `HBV_temp`: Input temperature (°C).
            - `HBV_prec`: Input precipitation (mm).
            - `HBV_rain`: Rainfall off-glacier (mm).
            - `HBV_snow`: Snowfall off-glacier (mm).
            - `HBV_pe`: Potential evapotranspiration (mm).
            - `HBV_snowpack`: Snowpack water equivalent (mm).
            - `HBV_soil_moisture`: Soil moisture content (mm).
            - `HBV_AET`: Actual evapotranspiration (mm).
            - `HBV_refreezing`: Refreezing within the snowpack (mm).
            - `HBV_upper_gw`: Water in the upper groundwater box (mm).
            - `HBV_lower_gw`: Water in the lower groundwater box (mm).
            - `HBV_melt_off_glacier`: Meltwater runoff from snow and refreezing off-glacier (mm).
            - `Q_HBV`: Simulated catchment runoff (mm).

    References
    ----------
    - Ayzel, G. (2016). LHMP: Lumped hydrological modeling playground.
      Zenodo. https://doi.org/10.5281/zenodo.59501
    - Oudin, L., Hervé, A., Perrin, C., Michel, C., Andréassian, V., Anctil, F., & Loumagne, C. (2005).
      Which potential evapotranspiration input for a lumped rainfall–runoff model?: Part 2—Towards a simple and efficient
      potential evapotranspiration model for rainfall–runoff modelling. *Journal of Hydrology, 303*(1), 290–306.
      https://doi.org/10.1016/j.jhydrol.2004.08.026

    Notes
    -----
    1. Calculates potential evapotranspiration using Oudin et al. (2005) if not provided.
    2. Simulates snowmelt, soil moisture recharge, evaporation, and groundwater flow iteratively over the setup and simulation periods.
    3. Supports optional scaling of snow and rain fractions based on dynamically changing glacier areas.
    """

    print("*-------------------*")
    print("Running HBV routine")
    # 1. new temporary dataframe from input with daily values
    if "PE" in input_df_catchment.columns:
        input_df_hbv = input_df_catchment.resample("D").agg(
            {"T2": "mean", "RRR": "sum", "rain": "sum", "snow": "sum", "PE": "sum"}
        )
    else:
        input_df_hbv = input_df_catchment.resample("D").agg(
            {"T2": "mean", "RRR": "sum", "rain": "sum", "snow": "sum"}
        )

    Temp = input_df_hbv["T2"]
    Prec = input_df_hbv["RRR"]
    rain = input_df_hbv["rain"]
    snow = input_df_hbv["snow"]

    # Calculation of PE with Oudin et al. 2005
    latent_heat_flux = 2.45
    water_density = 1000
    if "PE" in input_df_catchment.columns:
        Evap = input_df_hbv["PE"]
    else:
        doy = np.array(input_df_hbv.index.strftime("%j")).astype(int)
        lat = np.deg2rad(parameter.lat)

        # Part 2. Extraterrestrial radiation calculation
        # set solar constant (in W m-2)
        Rsc = 1367  # solar constant (in W m-2)
        # calculate solar declination dt (in radians)
        dt = 0.409 * np.sin(2 * np.pi / 365 * doy - 1.39)
        # calculate sunset hour angle (in radians)
        ws = np.arccos(-np.tan(lat) * np.tan(dt))
        # Calculate day angle j (in radians)
        j = 2 * np.pi / 365.25 * doy
        # Calculate relative distance to sun
        dr = 1.0 + 0.03344 * np.cos(j - 0.048869)
        # Calculate extraterrestrial radiation (J m-2 day-1)
        Re = (
            Rsc
            * 86400
            / np.pi
            * dr
            * (ws * np.sin(lat) * np.sin(dt) + np.sin(ws) * np.cos(lat) * np.cos(dt))
        )
        # convert from J m-2 day-1 to MJ m-2 day-1
        Re = Re / 10**6

        Evap = np.where(
            Temp + 5 > 0,
            (Re / (water_density * latent_heat_flux)) * ((Temp + 5) / 100) * 1000,
            0,
        )

        Evap = pd.Series(Evap, index=input_df_hbv.index)
        input_df_hbv["PE"] = Evap

    # 2. Set-up period:
    # 2.1 meteorological forcing preprocessing
    Temp_cal = Temp[parameter.set_up_start : parameter.set_up_end]
    Prec_cal = Prec[parameter.set_up_start : parameter.set_up_end]
    SNOW_cal = snow[parameter.set_up_start : parameter.set_up_end]
    RAIN_cal = rain[parameter.set_up_start : parameter.set_up_end]
    Evap_cal = Evap[parameter.set_up_start : parameter.set_up_end]

    # get the new glacier area for each year      --> I think this section is redundant. glacier_area does not cover the set_up period!
    if glacier_area is not None:

        glacier_area = glacier_area.iloc[1:, :].copy()
        SNOW2 = pd.DataFrame(SNOW_cal)
        SNOW2["area"] = 0
        for year in range(len(glacier_area)):
            SNOW2["area"] = np.where(
                SNOW2.index.year == glacier_area["time"].iloc[year],
                glacier_area["glacier_area"].iloc[year],
                SNOW2["area"],
            )

        SNOW2["snow"] = SNOW2["snow"] * (1 - (SNOW2["area"] / parameter.area_cat))
        SNOW_cal = SNOW2["snow"].squeeze()
        RAIN_cal = RAIN_cal * (1 - (SNOW2["area"] / parameter.area_cat))
        Prec_cal = Prec_cal * (1 - (SNOW2["area"] / parameter.area_cat))
    else:
        SNOW_cal = SNOW_cal * (1 - (parameter.area_glac / parameter.area_cat))
        RAIN_cal = RAIN_cal * (1 - (parameter.area_glac / parameter.area_cat))
        Prec_cal = Prec_cal * (1 - (parameter.area_glac / parameter.area_cat))

    # evaporation correction
    # a. calculate long-term averages of daily temperature
    Temp_mean_cal = np.array(
        [Temp_cal.loc[Temp_cal.index.dayofyear == x].mean() for x in range(1, 367)]
    )
    # b. correction of Evaporation daily values
    Evap_cal = Evap_cal.index.map(
        lambda x: (1 + parameter.CET * (Temp_cal[x] - Temp_mean_cal[x.dayofyear - 1]))
        * Evap_cal[x]
    )
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
        melt = max(0, parameter.CFMAX_snow * Temp_cal[t])  # control melting
        melt = min(melt, SNOWPACK_cal[t])  # limit by snowpack

        # how meltwater box forms
        SNOWMELT_cal[t] = SNOWMELT_cal[t - 1] + melt

        # snowpack after melting
        SNOWPACK_cal[t] -= melt

        # refreezing accounting
        refreezing = (
            parameter.CFR * parameter.CFMAX_snow * (parameter.TT_snow - Temp_cal[t])
        )
        refreezing = max(0, refreezing)  # control refreezing
        refreezing = min(refreezing, SNOWMELT_cal[t])  # limit by meltwater

        # snowpack after refreezing
        SNOWPACK_cal[t] += refreezing

        # meltwater after refreezing
        SNOWMELT_cal[t] -= refreezing

        # recharge to soil
        tosoil = max(
            0, SNOWMELT_cal[t] - (parameter.CWH * SNOWPACK_cal[t])
        )  # control recharge

        # meltwater after recharge to soil
        SNOWMELT_cal[t] -= tosoil

        # 2.3.1 Soil and evaporation routine
        # soil wetness calculation
        soil_wetness = (SM_cal[t - 1] / parameter.FC) ** parameter.BETA
        soil_wetness = max(0, min(1, soil_wetness))  # control soil wetness

        # soil recharge
        recharge = (RAIN_cal[t] + tosoil) * soil_wetness

        # soil moisture update
        SM_cal[t] = SM_cal[t - 1] + RAIN_cal[t] + tosoil - recharge

        # excess of water calculation
        excess = max(0, SM_cal[t] - parameter.FC)  # control excess

        # soil moisture update
        SM_cal[t] -= excess

        # evaporation accounting
        evapfactor = SM_cal[t] / (parameter.LP * parameter.FC)
        evapfactor = max(0, min(1, evapfactor))  # control evapfactor in range [0, 1]

        # calculate actual evaporation
        ETact_cal[t] = min(
            SM_cal[t], Evap_cal[t] * evapfactor
        )  # control actual evaporation

        # last soil moisture updating
        SM_cal[t] -= ETact_cal[t]

    print("Finished spin up for initial HBV parameters")

    # 3. meteorological forcing preprocessing for simulation
    Temp = Temp[parameter.sim_start : parameter.sim_end]
    Prec = Prec[parameter.sim_start : parameter.sim_end]
    SNOW = snow[parameter.sim_start : parameter.sim_end]
    RAIN = rain[parameter.sim_start : parameter.sim_end]
    Evap = Evap[parameter.sim_start : parameter.sim_end]

    # get the new glacier area for each year
    if glacier_area is not None:
        glacier_area = glacier_area.iloc[1:, :].copy()
        SNOW2 = pd.DataFrame(SNOW)
        SNOW2["area"] = 0
        for year in range(len(glacier_area)):
            SNOW2["area"] = np.where(
                SNOW2.index.year == glacier_area["time"].iloc[year],
                glacier_area["glacier_area"].iloc[year],
                SNOW2["area"],
            )
        RAIN = RAIN * (1 - (SNOW2["area"] / parameter.area_cat))  # Rain off glacier
        SNOW2["snow"] = SNOW2["snow"] * (
            1 - (SNOW2["area"] / parameter.area_cat)
        )  # Snow off-glacier
        SNOW = SNOW2["snow"].squeeze()
        Prec = Prec * (1 - (SNOW2["area"] / parameter.area_cat))

    else:
        RAIN = RAIN * (
            1 - (parameter.area_glac / parameter.area_cat)
        )  # Rain off glacier
        SNOW = SNOW * (
            1 - (parameter.area_glac / parameter.area_cat)
        )  # Snow off-glacier
        Prec = Prec * (1 - (parameter.area_glac / parameter.area_cat))

    # a. calculate long-term averages of daily temperature
    Temp_mean = np.array(
        [Temp.loc[Temp.index.dayofyear == x].mean() for x in range(1, 367)]
    )
    # b. correction of Evaporation daily values
    Evap = Evap.index.map(
        lambda x: (1 + parameter.CET * (Temp[x] - Temp_mean[x.dayofyear - 1])) * Evap[x]
    )
    # c. control Evaporation
    Evap = np.where(Evap > 0, Evap, 0)

    # 4. initialize boxes and initial conditions after calibration
    # snowpack box
    SNOWPACK = np.zeros(len(Prec))
    SNOWPACK[0] = SNOWPACK_cal[-1]
    # meltwater box
    SNOWMELT = np.zeros(len(Prec))
    SNOWMELT[0] = 0.0001
    # total melt off-glacier --> added to write daily output
    off_glac = np.zeros(len(Prec))
    off_glac[0] = 0.0001
    # refreezing  --> added to write daily output
    refreezing_off_glacier = np.zeros(len(Prec))
    refreezing_off_glacier[0] = 0.0001
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
        melt = max(0, parameter.CFMAX_snow * Temp[t])  # control melting
        melt = min(melt, SNOWPACK[t])  # limit by snowpack

        # how meltwater box forms
        SNOWMELT[t] = SNOWMELT[t - 1] + melt

        # snowpack after melting
        SNOWPACK[t] -= melt

        # refreezing accounting
        refreezing = (
            parameter.CFR * parameter.CFMAX_snow * (parameter.TT_snow - Temp[t])
        )
        refreezing = max(0, refreezing)  # control refreezing
        refreezing = min(refreezing, SNOWMELT[t])  # limit by meltwater

        # write refreezing to output
        refreezing_off_glacier[t] = refreezing

        # snowpack after refreezing
        SNOWPACK[t] += refreezing

        # meltwater after refreezing
        SNOWMELT[t] -= refreezing

        # write total melt off-glacier to output
        off_glac[t] = max(melt - refreezing, 0)

        # recharge to soil
        tosoil = max(0, SNOWMELT[t] - (parameter.CWH * SNOWPACK[t]))  # control recharge

        # meltwater after recharge to soil
        SNOWMELT[t] -= tosoil

        # 5.2 Soil and evaporation routine
        # soil wetness calculation
        soil_wetness = (SM[t - 1] / parameter.FC) ** parameter.BETA
        soil_wetness = max(
            0, min(1, soil_wetness)
        )  # control soil wetness in range [0, 1]

        # soil recharge
        recharge = (RAIN[t] + tosoil) * soil_wetness

        # soil moisture update
        SM[t] = SM[t - 1] + RAIN[t] + tosoil - recharge

        # excess of water calculation
        excess = max(0, SM[t] - parameter.FC)  # control excess

        # soil moisture update
        SM[t] -= excess

        # evaporation accounting
        evapfactor = SM[t] / (parameter.LP * parameter.FC)
        evapfactor = max(0, min(1, evapfactor))  # control evapfactor in range [0, 1]

        # calculate actual evaporation
        ETact[t] = min(SM[t], Evap[t] * evapfactor)  # control actual evaporation

        # last soil moisture updating
        SM[t] -= ETact[t]

        # 5.3 Groundwater routine
        # upper groundwater box
        SUZ[t] = SUZ[t - 1] + recharge + excess

        # percolation control
        perc = min(SUZ[t], parameter.PERC)

        # update upper groundwater box
        SUZ[t] -= perc

        # runoff from the highest part of upper groundwater box (surface runoff)
        Q0 = parameter.K0 * max(SUZ[t] - parameter.UZL, 0)

        # update upper groundwater box
        SUZ[t] -= Q0

        # runoff from the middle part of upper groundwater box
        Q1 = parameter.K1 * SUZ[t]

        # update upper groundwater box
        SUZ[t] -= Q1

        # calculate lower groundwater box
        SLZ[t] = SLZ[t - 1] + perc

        # runoff from lower groundwater box
        Q2 = parameter.K2 * SLZ[t]

        # update lower groundwater box
        SLZ[t] -= Q2

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
        {
            "HBV_temp": Temp,
            "HBV_prec": Prec,
            "HBV_rain": RAIN,
            "HBV_snow": SNOW,
            "HBV_pe": Evap,
            "HBV_snowpack": SNOWPACK,
            "HBV_soil_moisture": SM,
            "HBV_AET": ETact,
            "HBV_refreezing": refreezing_off_glacier,
            "HBV_upper_gw": SUZ,
            "HBV_lower_gw": SLZ,
            "HBV_melt_off_glacier": off_glac,
            "Q_HBV": Qsim,
        },
        index=input_df_hbv[parameter.sim_start : parameter.sim_end].index,
    )
    print("Finished HBV routine")
    return hbv_results


def create_statistics(output_MATILDA):
    """
    Generates descriptive statistics for a given MATILDA output DataFrame.

    Parameters
    ----------
    output_MATILDA : pandas.DataFrame
        Input DataFrame containing MATILDA model output variables.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing descriptive statistics (e.g., mean, standard deviation, min, max)
        for each column in `output_MATILDA`, along with the sum of all columns appended as an additional row.

    Notes
    -----
    1. The sum row is labeled as "sum" and included at the bottom of the statistics DataFrame.
    2. All values are rounded to three decimal places for consistency.
    """

    stats = output_MATILDA.describe()
    column_sums = pd.DataFrame(
        output_MATILDA.sum()
    )  # Renamed from `sum` to `column_sums`
    column_sums.columns = ["sum"]
    column_sums = column_sums.transpose()
    stats = pd.concat([stats, column_sums])
    stats = stats.round(3)
    return stats


def matilda_submodules(
    df_preproc,
    parameter,
    obs=None,
    glacier_profile=None,
    elev_rescaling=False,
    drop_surplus=False,
):
    """
    Executes the main MATILDA simulation, which integrates data preprocessing, Degree Day Model (DDM), HBV hydrological model,
    and optional glacier elevation rescaling to produce comprehensive runoff and hydrological outputs.

    Parameters
    ----------
    df_preproc : pandas.DataFrame
        Preprocessed input dataset containing climate variables such as temperature, precipitation, snow, and rain.
    parameter : pandas.Series
        Series of MATILDA parameters including:
            - Simulation settings (`sim_start`, `sim_end`, `set_up_start`, `set_up_end`, `freq`, `freq_long`).
            - Glacier parameters (`area_glac`, `area_cat`, `ele_glac`, `ele_cat`, `warn`).
            - HBV and DDM model parameters.
    obs : pandas.DataFrame, optional
        Observed runoff data for model evaluation. Defaults to None.
    glacier_profile : pandas.DataFrame, optional
        Initial glacier profile with columns such as `Area`, `WE` (water equivalent), and `Elevation`. Required if `elev_rescaling=True`.
    elev_rescaling : bool, optional
        If True, enables glacier elevation rescaling based on the deltaH scaling routine. Defaults to False.
    drop_surplus : bool, optional
        If True, drops runs where the cumulative surface mass balance (SMB) is positive. Defaults to False.

    Returns
    -------
    list
        A list containing the following elements:
            - `output_MATILDA_compact` (pandas.DataFrame): Compact output with key variables for quick assessment.
            - `output_MATILDA` (pandas.DataFrame): Full simulation results including all variables.
            - `kge` (str or float): Kling-Gupta Efficiency coefficient for model performance evaluation.
            - `stats` (pandas.DataFrame): Statistical summary of the simulation results.
            - `lookup_table` (str or pandas.DataFrame): Lookup table for glacier area changes (if applicable).
            - `glacier_change` (str or pandas.DataFrame): Time series of glacier changes (if applicable).

    Notes
    -----
    1. If `elev_rescaling=True` and `glacier_profile` is provided, glacier elevation and area are dynamically updated annually.
    2. When glacier elevation scaling is turned off, average glacier elevation is treated as constant, which may introduce biases for longer simulations.
    3. Observed runoff (`obs`) is required for model efficiency metrics such as KGE, NSE, RMSE, and MARE.
    4. The function calculates and outputs both compact and detailed results for downstream analyses.

    Warnings
    --------
    - If no glacier profile is provided while `elev_rescaling=True`, an error is raised.
    - Glacier melt calculations are skipped if the glacier area is zero (`area_glac = 0`).
    - Model efficiency metrics are not calculated if observed data (`obs`) is not provided.
    """

    # Filter warnings:
    if not parameter.warn:
        warnings.filterwarnings(action="ignore")

    print("---")
    print("Initiating MATILDA simulation")

    # Rescale glacier elevation or not?
    if elev_rescaling:
        # Execute glacier-rescaling module
        if parameter.area_glac > 0:
            if glacier_profile is not None:
                lookup_table = create_lookup_table(glacier_profile, parameter)
                output_DDM, glacier_change, input_df_catchment = updated_glacier_melt(
                    df_preproc,
                    lookup_table,
                    glacier_profile,
                    parameter,
                    drop_surplus=drop_surplus,
                )
            else:
                raise ValueError(
                    "ERROR: No glacier profile passed for glacier elevation rescaling! Provide a glacier profile or"
                    " set elev_rescaling=False"
                )
        else:
            lookup_table = str("No lookup table generated")
            glacier_change = str("No glacier changes calculated")

    else:
        print(
            "WARNING: Glacier elevation scaling is turned off. The average glacier elevation is treated as constant. "
            "This might cause a significant bias in glacier melt on larger time scales! Set elev_rescaling=True "
            "to annually rescale glacier elevations."
        )
        # Scale input data to fit catchments elevations
        if parameter.ele_dat is not None:
            input_df_glacier, input_df_catchment = input_scaling(df_preproc, parameter)
        else:
            input_df_glacier = df_preproc.copy()
            input_df_catchment = df_preproc.copy()

        input_df_glacier = input_df_glacier[parameter.sim_start : parameter.sim_end]

        # Execute DDM module
        if parameter.area_glac > 0:
            degreedays_ds = calculate_PDD(input_df_glacier)
            output_DDM = calculate_glaciermelt(degreedays_ds, parameter)

        # Execute glacier re-scaling module
        if parameter.area_glac > 0:
            if glacier_profile is not None:
                lookup_table = create_lookup_table(glacier_profile, parameter)
                output_DDM, glacier_change = glacier_area_change(
                    output_DDM, lookup_table, glacier_profile, parameter
                )
            else:
                # scaling DDM output to fraction of catchment area
                for col in output_DDM.columns.drop(["DDM_smb", "pdd"]):
                    output_DDM[col + "_scaled"] = output_DDM[col] * (
                        parameter.area_glac / parameter.area_cat
                    )

                lookup_table = str("No lookup table generated")
                glacier_change = str("No glacier changes calculated")
        else:
            lookup_table = str("No lookup table generated")
            glacier_change = str("No glacier changes calculated")

    # Execute HBV module:
    if glacier_profile is not None:
        output_HBV = hbv_simulation(
            input_df_catchment, parameter, glacier_area=glacier_change
        )
    else:
        output_HBV = hbv_simulation(input_df_catchment, parameter)
    output_HBV = output_HBV[parameter.sim_start : parameter.sim_end]

    # Output postprocessing
    if parameter.area_glac > 0:
        output_MATILDA = pd.concat([output_HBV, output_DDM], axis=1)
    else:
        output_MATILDA = output_HBV.copy()

    if obs is not None:
        output_MATILDA = pd.concat([output_MATILDA, obs], axis=1)

    if parameter.area_glac > 0:
        if glacier_profile is not None:
            output_MATILDA["Q_Total"] = (
                output_MATILDA["Q_HBV"] + output_MATILDA["Q_DDM_updated_scaled"]
            )
            output_MATILDA["Prec_total"] = (
                output_MATILDA["DDM_rain_updated_scaled"]
                + output_MATILDA["DDM_snow_updated_scaled"]
                + output_MATILDA["HBV_rain"]
                + output_MATILDA["HBV_snow"]
            )
            output_MATILDA["Melt_total"] = (
                output_MATILDA["DDM_total_melt_updated_scaled"]
                + output_MATILDA["HBV_melt_off_glacier"]
            )
        else:
            output_MATILDA["Q_Total"] = (
                output_MATILDA["Q_HBV"] + output_MATILDA["Q_DDM_scaled"]
            )
            output_MATILDA["Prec_total"] = (
                output_MATILDA["DDM_rain_scaled"]
                + output_MATILDA["DDM_snow_scaled"]
                + output_MATILDA["HBV_rain"]
                + output_MATILDA["HBV_snow"]
            )
            output_MATILDA["Melt_total"] = (
                output_MATILDA["DDM_total_melt_scaled"]
                + output_MATILDA["HBV_melt_off_glacier"]
            )
    else:
        output_MATILDA["Q_Total"] = output_MATILDA["Q_HBV"]

    output_MATILDA = output_MATILDA[parameter.sim_start : parameter.sim_end]

    if "smb_flag" in output_MATILDA.columns:
        output_MATILDA["Q_Total"] = 0.01

    # Add compact output
    if parameter.area_glac > 0:
        if glacier_profile is not None:
            output_MATILDA_compact = pd.DataFrame(
                {
                    "avg_temp_catchment": output_MATILDA["HBV_temp"],
                    "avg_temp_glaciers": output_MATILDA["DDM_temp"],
                    "evap_off_glaciers": output_MATILDA["HBV_pe"],
                    "prec_off_glaciers": output_MATILDA["HBV_prec"],
                    "prec_on_glaciers": output_MATILDA["DDM_prec_updated_scaled"],
                    "rain_off_glaciers": output_MATILDA["HBV_rain"],
                    "snow_off_glaciers": output_MATILDA["HBV_snow"],
                    "rain_on_glaciers": output_MATILDA["DDM_rain_updated_scaled"],
                    "snow_on_glaciers": output_MATILDA["DDM_snow_updated_scaled"],
                    "snowpack_off_glaciers": output_MATILDA["HBV_snowpack"],
                    "soil_moisture": output_MATILDA["HBV_soil_moisture"],
                    "upper_groundwater": output_MATILDA["HBV_upper_gw"],
                    "lower_groundwater": output_MATILDA["HBV_lower_gw"],
                    "melt_off_glaciers": output_MATILDA["HBV_melt_off_glacier"],
                    "melt_on_glaciers": output_MATILDA["DDM_total_melt_updated_scaled"],
                    "ice_melt_on_glaciers": output_MATILDA[
                        "DDM_ice_melt_updated_scaled"
                    ],
                    "snow_melt_on_glaciers": output_MATILDA[
                        "DDM_snow_melt_updated_scaled"
                    ],
                    "refreezing_glacier": output_MATILDA[
                        "DDM_refreezing_updated_scaled"
                    ],
                    "total_refreezing": output_MATILDA["DDM_refreezing_updated_scaled"]
                    + output_MATILDA["HBV_refreezing"],
                    "SMB": output_MATILDA["DDM_smb"],
                    "actual_evaporation": output_MATILDA["HBV_AET"],
                    "total_precipitation": output_MATILDA["Prec_total"],
                    "total_melt": output_MATILDA["Melt_total"],
                    "runoff_without_glaciers": output_MATILDA["Q_HBV"],
                    "runoff_from_glaciers": output_MATILDA["Q_DDM_updated_scaled"],
                    "runoff_ratio": np.where(
                        output_MATILDA["Prec_total"] == 0,
                        0,
                        output_MATILDA["Q_Total"] / output_MATILDA["Prec_total"],
                    ),
                    "total_runoff": output_MATILDA["Q_Total"],
                },
                index=output_MATILDA.index,
            )
            if obs is not None:
                output_MATILDA_compact["observed_runoff"] = output_MATILDA["Qobs"]
        else:
            output_MATILDA_compact = pd.DataFrame(
                {
                    "avg_temp_catchment": output_MATILDA["HBV_temp"],
                    "avg_temp_glaciers": output_MATILDA["DDM_temp"],
                    "evap_off_glaciers": output_MATILDA["HBV_pe"],
                    "prec_off_glaciers": output_MATILDA["HBV_prec"],
                    "prec_on_glaciers": output_MATILDA["DDM_prec_scaled"],
                    "rain_off_glaciers": output_MATILDA["HBV_rain"],
                    "snow_off_glaciers": output_MATILDA["HBV_snow"],
                    "rain_on_glaciers": output_MATILDA["DDM_rain_scaled"],
                    "snow_on_glaciers": output_MATILDA["DDM_snow_scaled"],
                    "snowpack_off_glaciers": output_MATILDA["HBV_snowpack"],
                    "soil_moisture": output_MATILDA["HBV_soil_moisture"],
                    "upper_groundwater": output_MATILDA["HBV_upper_gw"],
                    "lower_groundwater": output_MATILDA["HBV_lower_gw"],
                    "melt_off_glaciers": output_MATILDA["HBV_melt_off_glacier"],
                    "melt_on_glaciers": output_MATILDA["DDM_total_melt_scaled"],
                    "ice_melt_on_glaciers": output_MATILDA["DDM_ice_melt_scaled"],
                    "snow_melt_on_glaciers": output_MATILDA["DDM_snow_melt_scaled"],
                    "refreezing_glacier": output_MATILDA["DDM_refreezing_scaled"],
                    "total_refreezing": output_MATILDA["DDM_refreezing_scaled"]
                    + output_MATILDA["HBV_refreezing"],
                    "SMB": output_MATILDA["DDM_smb"],
                    "actual_evaporation": output_MATILDA["HBV_AET"],
                    "total_precipitation": output_MATILDA["Prec_total"],
                    "total_melt": output_MATILDA["Melt_total"],
                    "runoff_without_glaciers": output_MATILDA["Q_HBV"],
                    "runoff_from_glaciers": output_MATILDA["Q_DDM_scaled"],
                    "runoff_ratio": np.where(
                        output_MATILDA["Prec_total"] == 0,
                        0,
                        output_MATILDA["Q_Total"] / output_MATILDA["Prec_total"],
                    ),
                    "total_runoff": output_MATILDA["Q_Total"],
                },
                index=output_MATILDA.index,
            )
            if obs is not None:
                output_MATILDA_compact["observed_runoff"] = output_MATILDA["Qobs"]

    else:
        output_MATILDA_compact = pd.DataFrame(
            {
                "avg_temp_catchment": output_MATILDA["HBV_temp"],
                "prec": output_MATILDA["HBV_prec"],
                "rain": output_MATILDA["HBV_rain"],
                "snow": output_MATILDA["HBV_snow"],
                "snowpack": output_MATILDA["HBV_snowpack"],
                "soil_moisture": output_MATILDA["HBV_soil_moisture"],
                "upper_groundwater": output_MATILDA["HBV_upper_gw"],
                "lower_groundwater": output_MATILDA["HBV_lower_gw"],
                "snow_melt": output_MATILDA["HBV_melt_off_glacier"],
                "total_refreezing": output_MATILDA["HBV_refreezing"],
                "actual_evaporation": output_MATILDA["HBV_AET"],
                "runoff_ratio": np.where(
                    output_MATILDA["HBV_prec"] == 0,
                    0,
                    output_MATILDA["Q_HBV"] / output_MATILDA["HBV_prec"],
                ),
                "runoff": output_MATILDA["Q_HBV"],
            },
            index=output_MATILDA.index,
        )
        if obs is not None:
            output_MATILDA_compact["observed_runoff"] = output_MATILDA["Qobs"]

        # if obs is not None:
    #     output_MATILDA.loc[output_MATILDA.isnull().any(axis=1), :] = np.nan

    # Model efficiency coefficients
    if obs is not None and "smb_flag" not in output_MATILDA.columns:
        sim = output_MATILDA["Q_Total"]
        target = output_MATILDA["Qobs"]
        # Crop both timeseries to same periods without NAs
        sim_new = pd.DataFrame()
        sim_new["mod"] = pd.DataFrame(sim)
        sim_new["obs"] = target
        clean = sim_new.dropna()
        sim = clean["mod"]
        target = clean["obs"]

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
            print(
                "** Model efficiency based on " + parameter.freq_long + " aggregates **"
            )
            print("KGE coefficient: " + str(round(float(kge), 2)))
            print("NSE coefficient: " + str(round(nash_sut, 2)))
            print("RMSE: " + str(round(float(rmse), 2)))
            print("MARE coefficient: " + str(round(float(mare), 2)))
            print("*-------------------*")
    else:
        kge = str(
            "No observations available to calculate model efficiency coefficients."
        )

    # if obs is not None:
    dat_stats = output_MATILDA_compact.copy()
    dat_stats.loc[dat_stats.isnull().any(axis=1), :] = np.nan
    stats = create_statistics(dat_stats)

    print(stats)
    print("End of the MATILDA simulation")
    print("---")
    output_MATILDA = output_MATILDA.round(3)
    output_all = [
        output_MATILDA_compact,
        output_MATILDA,
        kge,
        stats,
        lookup_table,
        glacier_change,
    ]

    return output_all


def matilda_plots(output_MATILDA, parameter, plot_type="print"):
    """
    Generates visualizations for MATILDA simulation results, including meteorological inputs, runoff outputs, and HBV subdomain variables.
    Supports multiple plotting styles (static, interactive, and combined).

    Parameters
    ----------
    output_MATILDA : list
        MATILDA simulation output containing the following elements:
            - Compact output (pandas.DataFrame): Key simulation variables (e.g., runoff, precipitation, temperature).
            - Full simulation results (pandas.DataFrame).
            - Model efficiency metric (e.g., Kling-Gupta Efficiency coefficient).
            - Statistics (pandas.DataFrame).
            - Glacier lookup table (if applicable).
            - Glacier change data (if applicable).
    parameter : pandas.Series
        Series of MATILDA parameters, including:
            - `freq`: Frequency for resampling (e.g., "D" for daily, "M" for monthly, "Y" for yearly).
            - `freq_long`: Long-form frequency description (e.g., "Daily", "Monthly").
            - `sim_start`: Start date of the simulation period (YYYY-MM-DD).
            - `sim_end`: End date of the simulation period (YYYY-MM-DD).
    plot_type : str, optional
        Specifies the type of plots to generate:
            - `"print"`: Static plots using Matplotlib.
            - `"interactive"`: Interactive plots using Plotly.
            - `"all"`: Both static and interactive plots.
            Defaults to `"print"`.

    Returns
    -------
    list
        The updated `output_MATILDA` list with added visualizations. The plots are appended as follows:
            - Static Matplotlib plots: `[fig1, fig2, fig3]` for meteorological inputs, runoff, and HBV subdomains, respectively.
            - Interactive Plotly plots: `[fig1, fig2]` for full results and annual mean results, respectively.

    Notes
    -----
    1. **Static Plots**:
        - Plots meteorological inputs (temperature, precipitation, evaporation).
        - Visualizes runoff contributions and comparisons with observed data (if available).
        - Displays HBV subdomain outputs (soil moisture, snowpack, groundwater).
    2. **Interactive Plots**:
        - Provides detailed and zoomable visualizations for meteorological inputs, runoff, and contributions using Plotly.
        - Includes annotations for model efficiency metrics like KGE.
    3. Data is resampled to the specified frequency (`freq`) for consistent visualization across time steps.
    4. Observed runoff data (`observed_runoff`) is included if available in the input.

    Warnings
    --------
    - Ensure that `output_MATILDA` contains valid simulation results before calling this function.
    - Interactive plots require Plotly; ensure it is installed for `"interactive"` or `"all"` options.
    - Large datasets may cause performance issues with interactive plotting.
    """

    # resampling the output to the specified frequency
    def plot_data(output_MATILDA, parameter):
        if "observed_runoff" in output_MATILDA[0].columns:
            # obs = output_MATILDA[1]["Qobs"].resample(parameter.freq).agg(pd.DataFrame.sum, skipna=False)
            obs = (
                output_MATILDA[0]["observed_runoff"]
                .resample(parameter.freq)
                .agg(pd.Series.sum, min_count=1)
            )
        if "Q_DDM" in output_MATILDA[1].columns:
            plot_data = (
                output_MATILDA[0]
                .resample(parameter.freq)
                .agg(
                    {
                        "avg_temp_catchment": "mean",
                        "rain_off_glaciers": "sum",
                        "rain_on_glaciers": "sum",
                        "prec_off_glaciers": "sum",
                        "prec_on_glaciers": "sum",
                        "total_precipitation": "sum",
                        "evap_off_glaciers": "sum",
                        "melt_off_glaciers": "sum",
                        "melt_on_glaciers": "sum",
                        "ice_melt_on_glaciers": "sum",
                        "snow_melt_on_glaciers": "sum",
                        "runoff_without_glaciers": "sum",
                        "runoff_from_glaciers": "sum",
                        "total_runoff": "sum",
                        "actual_evaporation": "sum",
                        "snowpack_off_glaciers": "mean",
                        "refreezing_glacier": "sum",
                        "total_refreezing": "sum",
                        "soil_moisture": "mean",
                        "upper_groundwater": "mean",
                        "lower_groundwater": "mean",
                    },
                    skipna=False,
                )
            )
        else:
            plot_data = (
                output_MATILDA[0]
                .resample(parameter.freq)
                .agg(
                    {
                        "avg_temp_catchment": "mean",
                        "rain_off_glaciers": "sum",
                        "rain_on_glaciers": "sum",
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
                        "lower_groundwater": "mean",
                    },
                    skipna=False,
                )
            )
        if "observed_runoff" in output_MATILDA[0].columns:
            plot_data["observed_runoff"] = obs

        plot_annual_data = output_MATILDA[0].copy()
        plot_annual_data["month"] = plot_annual_data.index.month
        plot_annual_data["day"] = plot_annual_data.index.day
        plot_annual_data = plot_annual_data.groupby(["month", "day"]).mean()
        plot_annual_data["date"] = pd.date_range(
            parameter.sim_start, freq="D", periods=len(plot_annual_data)
        ).strftime("%Y-%m-%d")
        plot_annual_data = plot_annual_data.set_index(plot_annual_data["date"])
        plot_annual_data.index = pd.to_datetime(plot_annual_data.index)
        plot_annual_data["plot"] = 0
        if parameter.freq == "Y":
            plot_annual_data = plot_annual_data.resample("M").agg(
                pd.Series.sum, min_count=1
            )
        else:
            plot_annual_data = plot_annual_data.resample(parameter.freq).agg(
                pd.Series.sum, min_count=1
            )

        return plot_data, plot_annual_data

    # Plot the meteorological variables
    def plot_meteo(plot_data, parameter):
        fig, (ax1, ax2, ax3) = plt.subplots(
            3, sharex=True, figsize=(8, 4.8), dpi=150, constrained_layout=True
        )

        x_vals = plot_data.index.to_pydatetime()
        plot_length = len(x_vals)
        ax1.plot(x_vals, (plot_data["avg_temp_catchment"]), c="#d7191c")
        ax1.set_xlim(x_vals[0], x_vals[-1])
        if plot_length > (365 * 5):
            # bar chart has very poor performance for large data sets -> switch to line chart
            ax2.fill_between(
                x_vals,
                plot_data["total_precipitation"],
                plot_data["prec_off_glaciers"],
                color="#77bbff",
            )
            ax2.fill_between(x_vals, plot_data["prec_off_glaciers"], 0, color="#3594dc")
        else:
            ax2.bar(
                x_vals,
                plot_data["prec_off_glaciers"],
                width=15,
                color="#8DC5FF",
                label="off glaciers",
            )
            ax2.bar(
                x_vals,
                plot_data["prec_on_glaciers"],
                width=15,
                color="#2E6EAE",
                label="on glaciers",
                bottom=plot_data["prec_off_glaciers"],
            )
            ax2.legend(prop={"size": 6})
        ax3.bar(x_vals, plot_data["evap_off_glaciers"], width=15, color="#008837")
        plt.xlabel("Date")
        ax1.grid(lw=0.25, ls=":"), ax2.grid(lw=0.25, ls=":"), ax3.grid(lw=0.25, ls=":")
        ax3.sharey(ax2)
        ax1.set_title("Mean temperature")
        ax2.set_title("Precipitation")
        ax3.set_title("Potential evapotranspiration")
        ax1.set_ylabel(r"[$\degree$C]")
        ax2.set_ylabel("[mm]")
        ax3.set_ylabel("[mm]")
        if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
            fig.suptitle(
                parameter.freq_long
                + " meteorological forcing ("
                + str(plot_data.index.values[-1])[:4]
                + ")"
            )
        else:
            fig.suptitle(
                parameter.freq_long
                + " meteorological forcing ("
                + str(plot_data.index.values[0])[:4]
                + "-"
                + str(plot_data.index.values[-1])[:4]
                + ")"
            )
        return fig

    # Plot the runoff
    def plot_runoff(plot_data, plot_annual_data, parameter):
        plot_data["plot"] = 0
        fig = plt.figure(figsize=(8, 5.5), dpi=150)

        left, right, bottom, top = [0.07, 0.98, 0.05, 0.95]  # margins
        gs = fig.add_gridspec(
            nrows=2,
            ncols=2,
            width_ratios=[2.75, 1],
            left=left,
            right=right,
            bottom=bottom,
            top=top,
            wspace=0.05,
            hspace=0.28,
        )

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharey=ax1)
        ax3 = fig.add_subplot(gs[2])
        ax4 = fig.add_subplot(gs[3], sharey=ax3)
        plt.setp(ax2.get_yticklabels(), visible=False)  # shared y -> no ticks
        plt.setp(ax4.get_yticklabels(), visible=False)  # shared y -> no ticks

        # AX1 (upper left): simulation vs observed for defined period
        x_vals = plot_data.index.to_pydatetime()
        if "observed_runoff" in plot_data.columns:
            ax1.plot(
                x_vals,
                plot_data["observed_runoff"],
                c="#E69F00",
                label="Observations",
                linewidth=1.2,
            )

        if "total_runoff" in plot_data.columns:
            # ax1.plot(x_vals, plot_data["total_runoff"], c="k", label="", linewidth=0.75, alpha=0.75)
            ax1.fill_between(
                x_vals,
                plot_data["runoff_without_glaciers"],
                plot_data["total_runoff"],
                color="#0C5DA5",
                edgecolor=None,
                alpha=0.75,
                label="Simulated runoff from glaciers",
            )
        ax1.fill_between(
            x_vals,
            plot_data["plot"],
            plot_data["runoff_without_glaciers"],
            color="#00B945",
            edgecolor=None,
            alpha=0.75,
            label="Simulated runoff off glaciers",
        )
        ax1.set_ylabel("Runoff [mm]")
        ax1.legend(loc="upper left", ncol=3, prop={"size": 6}, bbox_to_anchor=(0.01, 1))
        ax1.set_xlim(x_vals[0], x_vals[-1])

        if isinstance(output_MATILDA[2], float):
            anchored_text = AnchoredText(
                "KGE coeff " + str(round(output_MATILDA[2], 2)), loc=1, frameon=False
            )
        elif "observed_runoff" not in plot_data.columns:
            anchored_text = AnchoredText(" ", loc=2, frameon=False)
        else:
            anchored_text = AnchoredText(
                "KGE coeff exceeds boundaries", loc=2, frameon=False
            )
        ax2.add_artist(anchored_text)

        # AX2 (upper right): simulation vs observed annual data
        x_vals = plot_annual_data.index.to_pydatetime()
        if "observed_runoff" in plot_annual_data.columns:
            ax2.plot(
                x_vals,
                plot_annual_data["observed_runoff"],
                c="#E69F00",
                label="Observations",
                linewidth=1.2,
            )

        if "total_runoff" in plot_annual_data.columns:
            # ax2.plot(x_vals, plot_annual_data["total_runoff"], c="k", label="Simulated total runoff",
            #          linewidth=0.75, alpha=0.75)
            ax2.fill_between(
                x_vals,
                plot_annual_data["runoff_without_glaciers"],
                plot_annual_data["total_runoff"],
                color="#0C5DA5",
                edgecolor=None,
                alpha=0.75,
                label="Simulated runoff from glaciers",
            )
        ax2.fill_between(
            x_vals,
            plot_annual_data["plot"],
            plot_annual_data["runoff_without_glaciers"],
            color="#00B945",
            edgecolor=None,
            alpha=0.75,
            label="Simulated runoff off glaciers",
        )

        if parameter.freq == "Y" or parameter.freq == "M":
            # formatting for 'monthly' looks strange, therefore default ticks (major+minor) will be overwritten
            ax2.set_xticks(x_vals[::], minor=True)  # minor tick for each month
            ax2.set_xticks(x_vals[::3])  # major tick every 3rd month
            ax2.set_xticklabels(plot_annual_data.index.strftime("%b")[::3])
        else:
            ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax2.set_xlim(x_vals[0], x_vals[-1])

        # AX3: runoff contribution for period
        rain = plot_data["rain_off_glaciers"] + plot_data["rain_on_glaciers"]
        snow_melt = plot_data["melt_off_glaciers"] + plot_data["snow_melt_on_glaciers"]
        glacier_melt = plot_data["ice_melt_on_glaciers"]

        ax3.set_title("Runoff Contribution")

        x_vals = plot_data.index.to_pydatetime()
        ax3.set_xlim(x_vals[0], x_vals[-1])

        ax3.axhline(0, color="k", linewidth=0.5)
        ax3.stackplot(
            x_vals,
            rain,
            snow_melt,
            glacier_melt,
            labels=["Rain", "Snow melt", "Glacier melt"],
            colors=["#6c1e58", "#a6135a", "#d72f41"],
        )

        ax3.stackplot(
            x_vals,
            plot_data["actual_evaporation"] * -1,
            plot_data["total_refreezing"] * -1,
            labels=["Actual evaporation", "Refreezing"],
            colors=["#cce4ee", "#bababa"],
        )

        ax3.legend(loc="upper left", ncol=5, prop={"size": 6}, bbox_to_anchor=(0.01, 1))
        ax3.set_ylabel("[mm]")

        # AX4:
        x_vals = plot_annual_data.index.to_pydatetime()
        rain = (
            plot_annual_data["rain_off_glaciers"] + plot_annual_data["rain_on_glaciers"]
        )
        snow_melt = (
            plot_annual_data["melt_off_glaciers"]
            + plot_annual_data["snow_melt_on_glaciers"]
        )
        glacier_melt = plot_annual_data["ice_melt_on_glaciers"]

        ax4.axhline(0, color="k", linewidth=0.5)
        ax4.stackplot(
            x_vals,
            rain,
            snow_melt,
            glacier_melt,
            labels=["Rain", "Snow melt", "Glacier melt"],
            colors=["#6c1e58", "#a6135a", "#d72f41"],
        )

        ax4.stackplot(
            x_vals,
            plot_annual_data["actual_evaporation"] * -1,
            plot_annual_data["total_refreezing"] * -1,
            labels=["Actual evaporation", "Refreezing"],
            colors=["#cce4ee", "#bababa"],
        )

        if parameter.freq == "Y" or parameter.freq == "M":
            # formatting for 'monthly' looks strange, therefore default ticks (major+minor) will be overwritten
            ax4.set_xticks(x_vals[::], minor=True)  # minor tick for each month
            ax4.set_xticks(x_vals[::3])  # major tick every 3rd month
            ax4.set_xticklabels(plot_annual_data.index.strftime("%b")[::3])
        else:
            ax4.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            ax4.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        ax4.set_xlim(x_vals[0], x_vals[-1])

        # sync y-axis limits
        y_min1, y_max1 = ax1.dataLim.intervaly
        y_min3, y_max3 = ax3.dataLim.intervaly
        y_min = min(y_min1, y_min3)
        y_max = max(y_max1, y_max3)
        y_ext = (
            abs(y_min) + abs(y_max)
        ) / ax1.bbox.height  # normalize y-axis steps in pixels
        ax1.set_ylim(0, y_max + y_ext * 30)  # extent 30px pos (legend space)
        ax3.set_ylim(
            y_min - y_ext * 10, y_max + y_ext * 30
        )  # extent 10px neg and 30px pos (legend space)

        if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
            ax1.set_title(
                parameter.freq_long
                + " MATILDA simulation for the period "
                + str(plot_data.index.values[-1])[:4]
            )
        else:
            ax1.set_title(
                parameter.freq_long
                + " MATILDA simulation for the period "
                + str(plot_data.index.values[0])[:4]
                + "-"
                + str(plot_data.index.values[-1])[:4]
            )
        return fig

    # Plot the HBV output variables
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
        ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel(
            "[mm]", fontsize=9
        ), ax3.set_ylabel("[mm]", fontsize=9)
        ax4.set_ylabel("[mm]", fontsize=9), ax5.set_ylabel("[mm]", fontsize=9)
        if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
            fig.suptitle(
                parameter.freq_long
                + " output from the HBV model in the period "
                + str(plot_data.index.values[-1])[:4],
                size=14,
            )
        else:
            fig.suptitle(
                parameter.freq_long
                + " output from the HBV model in the period "
                + str(plot_data.index.values[0])[:4]
                + "-"
                + str(plot_data.index.values[-1])[:4],
                size=14,
            )
        plt.tight_layout()
        fig.set_size_inches(10, 6)
        return fig

    # Plot the meteorological variables with Plotly
    def plot_plotly_meteo(plot_data, fig, row):
        x_vals = plot_data.index.to_pydatetime()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["avg_temp_catchment"],
                name="Mean temperature",
                line_color="#d7191c",
                legendgroup="meteo",
                legendgrouptitle_text="Meteo",
            ),
            row=row,
            col=1,
            secondary_y=False,
        )
        # fig.add_trace(
        #     go.Bar(x=x_vals, y=plot_data["total_precipitation"], name="Precipitation sum", marker_color="#2c7bb6",
        #                legendgroup="meteo",  offsetgroup=0),
        #     row=row, col=1, secondary_y=True)
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=plot_data["prec_off_glaciers"],
                name="Precipitation off glacier",
                marker_color="#3594dc",
                legendgroup="meteo",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=plot_data["prec_on_glaciers"],
                name="Precipitation on glacier",
                marker_color="#77bbff",
                legendgroup="meteo",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )
        fig.add_trace(
            go.Bar(
                x=x_vals,
                y=plot_data["evap_off_glaciers"] * -1,
                name="Pot. evapotranspiration",
                marker_color="#008837",
                legendgroup="meteo",
            ),
            row=row,
            col=1,
            secondary_y=True,
        )

    # Plot the runoff/refreezing output variables with Plotly
    def plot_plotly_runoff(plot_data, fig, row):
        x_vals = plot_data.index.to_pydatetime()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["runoff_without_glaciers"],
                name="Simulated runoff off glaciers",
                fillcolor="#00B945",
                legendgroup="runoff",
                legendgrouptitle_text="Runoff comparison",
                stackgroup="one",
                mode="none",
            ),
            row=row,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["runoff_from_glaciers"],
                name="Simulated runoff from glaciers",
                fillcolor="#0C5DA5",
                legendgroup="runoff",
                stackgroup="one",
                mode="none",
            ),
            row=row,
            col=1,
        )

        if "observed_runoff" in plot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=plot_data["observed_runoff"],
                    name="Observations",
                    line_color="#E69F00",
                    legendgroup="runoff",
                ),
                row=row,
                col=1,
            )
        if "total_runoff" in plot_data.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_vals,
                    y=plot_data["total_runoff"],
                    name="MATILDA total runoff",
                    line_color="black",
                    legendgroup="runoff",
                ),
                row=row,
                col=1,
            )

        # add coefficient to plot
        if not isinstance(output_MATILDA[2], str):
            fig.add_annotation(
                xref="x domain",
                yref="y domain",
                x=0.99,
                y=0.95,
                xanchor="right",
                showarrow=False,
                text="<b>KGE coeff " + str(round(output_MATILDA[2], 2)) + "</b>",
                row=row,
                col=1,
            )

    # Plot the HBV output variables with Plotly
    def plot_plotly_hbv(plot_data, fig, row):
        x_vals = plot_data.index.to_pydatetime()
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["actual_evaporation"],
                name="Actual evapotranspiration",
                line_color="#16425b",
                legendgroup="hbv",
                legendgrouptitle_text="HBV subdomains",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["soil_moisture"],
                name="Soil moisture",
                line_color="#d9dcd6",
                legendgroup="hbv",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["snowpack_off_glaciers"],
                name="Water in snowpack",
                line_color="#81c3d7",
                legendgroup="hbv",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["upper_groundwater"],
                name="Upper groundwater box",
                line_color="#3a7ca5",
                legendgroup="hbv",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["lower_groundwater"],
                name="Lower groundwater box",
                line_color="#2f6690",
                legendgroup="hbv",
            ),
            row=row,
            col=1,
        )

    def plot_plotly_runoff_contrib(plot_data, fig, row):
        x_vals = plot_data.index.to_pydatetime()
        rain = plot_data["rain_off_glaciers"] + plot_data["rain_on_glaciers"]
        snow_melt = plot_data["melt_off_glaciers"] + plot_data["snow_melt_on_glaciers"]
        glacier_melt = plot_data["ice_melt_on_glaciers"]

        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=rain,
                name="Rain",
                fillcolor="#6c1e58",
                legendgroup="runoff2",
                legendgrouptitle_text="Runoff contributions",
                stackgroup="one",
                mode="none",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=snow_melt,
                name="Snow melt",
                fillcolor="#a6135a",
                legendgroup="runoff2",
                stackgroup="one",
                mode="none",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=glacier_melt,
                name="Glacier melt",
                fillcolor="#d72f41",
                legendgroup="runoff2",
                stackgroup="one",
                mode="none",
            ),
            row=row,
            col=1,
        )

        # negative values for act evaporation and refreezing
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["actual_evaporation"] * -1,
                name="Actual evaporation",
                fillcolor="#cce4ee",
                legendgroup="runoff2",
                stackgroup="two",
                mode="none",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=x_vals,
                y=plot_data["total_refreezing"] * -1,
                name="Refreezing",
                fillcolor="#bababa",
                legendgroup="runoff2",
                stackgroup="two",
                mode="none",
            ),
            row=row,
            col=1,
        )
        fig.add_hline(y=0, line_width=0.5, row=row, col=1)

    def plot_plotly(plot_data, plot_annual_data, parameter):
        # construct date range for chart titles
        range_from = str(plot_data.index.values[1])[:4]
        range_to = str(plot_data.index.values[-1])[:4]
        if range_from == range_to:
            date_range = range_from
        else:
            date_range = range_from + "-" + range_to
        title = [
            " Meteorological forcing data ",
            " Simulated vs runoff ",
            " Runoff contributions ",
            " HBV reservoirs ",
        ]
        title_f = []
        for i in range(len(title)):
            title_f.append("<b>" + title[i] + "</b>")

        # -- Plot 1 (combined charts) -- #
        # init plot
        fig1 = make_subplots(
            rows=4,
            cols=1,
            subplot_titles=title_f,
            shared_xaxes=True,
            vertical_spacing=0.15,
            specs=[
                [{"secondary_y": True}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
                [{"secondary_y": False}],
            ],
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
            plot_bgcolor="white",
            legend=dict(groupclick="toggleitem"),
            legend_tracegroupgap=20,
            xaxis_showticklabels=True,
            xaxis2_showticklabels=True,
            xaxis3_showticklabels=True,
            barmode="relative",
            hovermode="x",
            title={
                "text": parameter.freq_long + " MATILDA Results (" + date_range + ")",
                "font_size": 30,
                "x": 0.5,
                "xanchor": "center",
            },
            yaxis={"ticksuffix": " °C"},
            yaxis2={
                "ticksuffix": " mm",
            },
            yaxis3={"ticksuffix": " mm", "side": "right"},
            yaxis4={"ticksuffix": " mm", "side": "right"},
            yaxis5={"ticksuffix": " mm", "side": "right"},
        )

        # update x axes settings
        fig1.update_xaxes(
            ticks="outside",
            ticklabelmode="period",
            dtick="M12",
            tickcolor="black",
            tickwidth=2,
            ticklen=15,
            minor=dict(dtick="M1", ticklen=5),
        )

        # -- Plot 2 (annual mean) -- #
        title_annual = title[1:3]
        title_f = []
        for i in range(len(title_annual)):
            title_f.append("<b>" + title_annual[i] + "</b>")

        # init plot
        fig2 = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=title_f,
            shared_xaxes=True,
            vertical_spacing=0.15,
        )

        # Add subplot: MATILDA RUNOFF (annual data)
        plot_plotly_runoff(plot_annual_data, fig2, 1)

        # Add subplot: RUNOFF CONTRIBUTION (annual data)
        plot_plotly_runoff_contrib(plot_annual_data, fig2, 2)

        # update general layout settings
        fig2.update_layout(
            plot_bgcolor="white",
            legend=dict(groupclick="toggleitem"),
            legend_tracegroupgap=20,
            xaxis_showticklabels=True,
            hovermode="x",
            title={
                "text": "Annual mean MATILDA Results (" + date_range + ")",
                "font_size": 30,
                "x": 0.5,
                "xanchor": "center",
            },
            yaxis={"ticksuffix": " mm"},
            yaxis2={"ticksuffix": " mm"},
        )

        # update x axes settings
        fig2.update_xaxes(
            ticks="outside",
            ticklabelmode="period",
            dtick="M1",
            tickformat="%b",
            hoverformat="%d\n%b",
        )
        fig2.update_traces(marker={"opacity": 0})

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
    """
    Saves MATILDA simulation outputs, statistics, parameters, glacier area changes, and plots to the local disk.

    Parameters
    ----------
    output_MATILDA : list
        List containing the outputs from the MATILDA simulation. Includes:
            - Full model output (pandas.DataFrame).
            - Model statistics (pandas.DataFrame).
            - Model parameters (pandas.Series).
            - Glacier area data (optional, pandas.DataFrame).
            - Plots (Matplotlib or Plotly objects, depending on the plot type).
    parameter : pandas.Series
        Series containing the simulation parameters, including:
            - `sim_start`: Start date of the simulation period (YYYY-MM-DD).
            - `sim_end`: End date of the simulation period (YYYY-MM-DD).
    output_path : str
        Directory path where the output files will be saved. A subfolder with the simulation date and time will be created.
    plot_type : str, optional
        Specifies the type of plots to save:
            - `"print"`: Saves Matplotlib plots as PNG files.
            - `"interactive"`: Saves Plotly plots as HTML files.
            - `"all"`: Saves both Matplotlib and Plotly plots in their respective formats.
        Defaults to `"print"`.

    Returns
    -------
    None
        Saves the outputs and plots to the specified directory.

    Notes
    -----
    1. A subdirectory is created under `output_path` with a timestamped name for organizing the outputs.
    2. The outputs saved include:
        - Full simulation results as CSV (`model_output_<date_range>.csv`).
        - Statistics as CSV (`model_stats_<date_range>.csv`).
        - Parameters as CSV (`model_parameter.csv`).
        - Glacier area changes as CSV (`glacier_area_<date_range>.csv`) if applicable.
    3. Plots are saved in the specified format (`PNG` or `HTML`).
    4. The function automatically handles the appending of date ranges and timestamps to filenames.

    Warnings
    --------
    - Ensure the specified `output_path` exists and has write permissions.
    - When using `"interactive"` or `"all"`, ensure Plotly is installed for saving HTML plots.

    Examples
    --------
    Save MATILDA outputs and plots (Matplotlib only):
    >>> matilda_save_output(output_MATILDA, parameter, "/path/to/output", plot_type="print")

    Save MATILDA outputs and interactive plots (Plotly):
    >>> matilda_save_output(output_MATILDA, parameter, "/path/to/output", plot_type="interactive")

    Save MATILDA outputs with both static and interactive plots:
    >>> matilda_save_output(output_MATILDA, parameter, "/path/to/output", plot_type="all")
    """

    if output_path[-1] == "/":
        output_path = (
            output_path
            + parameter.sim_start[:4]
            + "_"
            + parameter.sim_end[:4]
            + "_"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "/"
        )
    else:
        output_path = (
            output_path
            + "/"
            + parameter.sim_start[:4]
            + "_"
            + parameter.sim_end[:4]
            + "_"
            + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            + "/"
        )
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
        output_MATILDA[6].savefig(
            output_path + "meteorological_data_" + date_range + ".png",
            bbox_inches="tight",
            dpi=output_MATILDA[6].dpi,
        )
        output_MATILDA[7].savefig(
            output_path + "model_runoff_" + date_range + ".png",
            dpi=output_MATILDA[7].dpi,
        )
        output_MATILDA[8].savefig(
            output_path + "HBV_output_" + date_range + ".png", dpi=output_MATILDA[8].dpi
        )

    elif plot_type == "interactive":
        # save plots from plotly as .html file
        output_MATILDA[6].write_html(
            output_path + "matilda_plots_" + date_range + ".html"
        )
        output_MATILDA[7].write_html(
            output_path + "matilda_plots_annual_" + date_range + ".html"
        )

    elif plot_type == "all":
        # save plots from matplotlib and plotly as .png files
        output_MATILDA[6].savefig(
            output_path + "meteorological_data_" + date_range + ".png",
            bbox_inches="tight",
            dpi=output_MATILDA[6].dpi,
        )
        output_MATILDA[7].savefig(
            output_path + "model_runoff_" + date_range + ".png",
            dpi=output_MATILDA[7].dpi,
        )
        output_MATILDA[8].savefig(
            output_path + "HBV_output_" + date_range + ".png", dpi=output_MATILDA[8].dpi
        )

        # save plots from plotly as .html file
        output_MATILDA[9].write_html(
            output_path + "matilda_plots_" + date_range + ".html"
        )
        output_MATILDA[10].write_html(
            output_path + "matilda_plots_annual_" + date_range + ".html"
        )

    print("---")


def matilda_simulation(
    input_df,
    obs=None,
    glacier_profile=None,
    output=None,
    warn=False,
    set_up_start=None,
    set_up_end=None,
    sim_start=None,
    sim_end=None,
    freq="D",
    lat=None,
    area_cat=None,
    area_glac=None,
    ele_dat=None,
    ele_glac=None,
    ele_cat=None,
    plots=True,
    plot_type="print",
    elev_rescaling=False,
    drop_surplus=False,
    **matilda_param,
):
    """
    Run the complete MATILDA simulation framework, including preprocessing, simulation, and optional output saving.

    This function integrates data preprocessing, hydrological and glacial simulations, and optional output handling
    using the MATILDA framework. It can perform simulations with default or user-defined parameters, handle optional
    glacier rescaling, and generate visualizations and model statistics.

    Parameters
    ----------
    input_df : pandas.DataFrame
        Input meteorological data for the simulation. Must include timestamp and relevant fields (e.g., temperature, precipitation).
    obs : pandas.DataFrame, optional
        Observed runoff data for model calibration or comparison. If provided, it will be included in outputs.
    glacier_profile : pandas.DataFrame, optional
        Glacier profile data for elevation-based glacier rescaling routines. If not provided, glacier rescaling is disabled.
    output : str, optional
        Directory path to save outputs (CSV files, statistics, plots). If not specified, outputs are not saved.
    warn : bool, optional
        Whether to show warnings during execution. Default is `False`.
    set_up_start : str, optional
        Start date for the model spin-up period (YYYY-MM-DD). Default is `None`.
    set_up_end : str, optional
        End date for the model spin-up period (YYYY-MM-DD). Default is `None`.
    sim_start : str, optional
        Start date for the simulation period (YYYY-MM-DD). Default is `None`.
    sim_end : str, optional
        End date for the simulation period (YYYY-MM-DD). Default is `None`.
    freq : str, optional
        Simulation time step frequency (`"D"` for daily, `"M"` for monthly). Default is `"D"`.
    lat : float, optional
        Latitude of the catchment area for potential evapotranspiration calculation. Required for HBV simulations.
    area_cat : float, optional
        Total catchment area in km². Required for runoff scaling.
    area_glac : float, optional
        Glacierized area of the catchment in km². Default is `None`.
    ele_dat : float, optional
        Elevation of the meteorological station used for input data in meters. Default is `None`.
    ele_glac : float, optional
        Average elevation of the glacierized area in meters. Default is `None`.
    ele_cat : float, optional
        Average elevation of the entire catchment in meters. Default is `None`.
    plots : bool, optional
        Whether to generate plots. Default is `True`.
    plot_type : str, optional
        Type of plots to generate. Options are `"print"`, `"interactive"`, or `"all"`. Default is `"print"`.
    elev_rescaling : bool, optional
        Whether to perform annual elevation rescaling for glaciers. Default is `False`.
    drop_surplus : bool, optional
        Whether to drop surplus glacial mass balance during simulation. Default is `False`.
    **matilda_param : dict, optional
        Additional model parameters passed as key-value pairs. These override the default parameter values.

    Returns
    -------
    list
        A list containing:
            - Compact MATILDA output (pandas.DataFrame).
            - Full MATILDA output (pandas.DataFrame).
            - Efficiency coefficient (str or float).
            - Model statistics (pandas.DataFrame).
            - Glacier lookup table (optional, str or pandas.DataFrame).
            - Glacier changes (optional, str or pandas.DataFrame).

    Notes
    -----
    1. Outputs are saved to disk if `output` is specified.
    2. Observed runoff data, if provided, is included in the efficiency calculations and final outputs.
    3. Plots can be suppressed by setting `plots=False`.
    """

    print("---")
    print("MATILDA framework")
    parameter = matilda_parameter(
        input_df,
        set_up_start=set_up_start,
        set_up_end=set_up_end,
        sim_start=sim_start,
        sim_end=sim_end,
        freq=freq,
        lat=lat,
        area_cat=area_cat,
        area_glac=area_glac,
        ele_dat=ele_dat,
        ele_glac=ele_glac,
        ele_cat=ele_cat,
        warn=warn,
        **matilda_param,
    )

    # Data preprocessing with the MATILDA preparation script
    if obs is None:
        df_preproc = matilda_preproc(input_df, parameter)
        # Downscaling of data if necessary and the MATILDA simulation
        if glacier_profile is not None:
            output_MATILDA = matilda_submodules(
                df_preproc,
                parameter,
                glacier_profile=glacier_profile,
                elev_rescaling=elev_rescaling,
                drop_surplus=drop_surplus,
            )
        else:
            output_MATILDA = matilda_submodules(df_preproc, parameter)
    else:
        df_preproc, obs_preproc = matilda_preproc(input_df, parameter, obs=obs)
        # Scale data if necessary and run the MATILDA simulation
        if glacier_profile is not None:
            output_MATILDA = matilda_submodules(
                df_preproc,
                parameter,
                obs=obs_preproc,
                glacier_profile=glacier_profile,
                elev_rescaling=elev_rescaling,
                drop_surplus=drop_surplus,
            )
        else:
            output_MATILDA = matilda_submodules(df_preproc, parameter, obs=obs_preproc)

    # Option to suppress plots.
    if plots:
        output_MATILDA = matilda_plots(output_MATILDA, parameter, plot_type)

    if output is not None:
        matilda_save_output(output_MATILDA, parameter, output, plot_type)

    return output_MATILDA
