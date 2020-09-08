# -*- coding: UTF-8 -*-
"""
The model is a combination between a degree day model and the HBV model (Bergstöm 1976) to compute runoff from the
glaciers and the catchment.
This file uses the input files created by COSIPY (aws2cosipy) as input files to run the model as well as additional
observation runoff data to validate it.
"""
##
import sys
sys.path.extend(['/home/ana/Seafile/SHK/Scripts/centralasiawaterresources/Final_Model'])
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import numpy as np
import scipy.signal as ss
import pandas as pd
import xarray as xr

from ConfigFile import *

print('---')
print('Read input netcdf file %s' % (cosipy_nc))
print('Read input csv file %s' % (cosipy_csv))
print('Read observation data %s' % (observation_data))

# Import necessary input: cosipy.nc, cosipy.csv and runoff observation data
# Observation data should be given in form of a csv with a date column and daily observations
ds = xr.open_dataset(input_path_cosipy + cosipy_nc)
df = pd.read_csv(input_path_cosipy + cosipy_csv)
obs = pd.read_csv(input_path_observations + observation_data)
if evap_data_available == True:
    evap = pd.read_csv(input_path_data + evap_data)

print("Adjust time period: " + str(time_start) + " until "  + str(time_end))

# adjust time
ds = ds.sel(time=slice(time_start, time_end))
df.set_index('TIMESTAMP', inplace=True)
df.index = pd.to_datetime(df.index)
df = df[time_start: time_end]
obs.set_index('Date', inplace=True)
obs.index = pd.to_datetime(obs.index)
# obs = obs.sort_index()
obs = obs[time_start: time_end]
if evap_data_available == True:
    evap.set_index("Date", inplace=True)
    evap.index = pd.to_datetime(evap.index)
    evap = evap[time_start: time_end]

## DDM
"""
Degree Day Model to calculate the accumulation, snow and ice melt and runoff rate from the glaciers.
Model input rewritten and adjusted to our needs from the pypdd function (github.com/juseg/pypdd 
- # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>)
"""
print("Running the degree day model")

def calculate_PDD(ds):
    temp_min = ds['T2'].resample(time="D").min(dim="time") # now °C
    temp_max = ds['T2'].resample(time="D").max(dim="time")
    temp_mean = ds['T2'].resample(time="D").mean(dim="time")
    if temp_unit:
        temp_max, temp_mean, temp_min = temp_max - 273.15, temp_mean - 273.15, temp_min - 273.15
    else:
        temp_max, temp_mean, temp_min = temp_max, temp_mean, temp_min

    prec = ds['RRR'].resample(time="D").sum(dim="time")
    time = temp_mean["time"]
    # masking the dataset to only get the glacier area
    mask = ds.MASK.values
    temp_min = xr.where(mask==1, temp_min, np.nan)
    temp_max = xr.where(mask==1, temp_max, np.nan)
    temp_mean = xr.where(mask==1, temp_mean, np.nan)
    prec = xr.where(mask==1, prec, np.nan)


    pdd_ds = xr.merge([xr.DataArray(temp_mean, name="temp_mean"), xr.DataArray(temp_min, name="temp_min"), \
                   xr.DataArray(temp_max, name="temp_max"), prec])

    # calculate the hydrological year
    def calc_hydrological_year(time):
        water_year = []
        for i in time:
            if 10 <= i["time.month"] <= 12:
                water_year.append(i["time.year"] + 1)
            else:
                water_year.append(i["time.year"])
        return np.asarray(water_year)

    #water_year = calc_hydrological_year(time)
    #pdd_ds = pdd_ds.assign_coords(water_year = water_year)

    # calculate the positive degree days
    pdd_ds["pdd"] = xr.where(temp_mean > 0, temp_mean, 0)

    return pdd_ds

degreedays_ds = calculate_PDD(ds)

# input from the pypdd model
def calculate_glaciermelt(ds):
    temp = ds["temp_mean"]
    prec = ds["RRR"]
    pdd = ds["pdd"]

    """ pypdd.py line 311
        Compute accumulation rate from temperature and precipitation.
        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.
    """
    reduced_temp = (parameters_DDM["temp_rain"] - temp) / (parameters_DDM["temp_rain"] - parameters_DDM["temp_snow"])
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
        pot_snow_melt = parameters_DDM['pdd_factor_snow'] * pdd
    # effective snow melt can't exceed amount of snow
        snow_melt = np.minimum(snow, pot_snow_melt)
    # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * parameters_DDM['pdd_factor_ice'] / parameters_DDM['pdd_factor_snow']
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
    runoff_rate = total_melt - parameters_DDM["refreeze_snow"] * snow_melt_rate \
                  - parameters_DDM["refreeze_ice"] * ice_melt_rate
    inst_smb = accu_rate - runoff_rate

    # making the final ds
    glacier_melt = xr.merge([xr.DataArray(accu_rate, name="accumulation_rate"), xr.DataArray(ice_melt_rate, \
                            name="ice_melt_rate"), xr.DataArray(snow_melt_rate, name="snow_melt_rate"), \
                             xr.DataArray(total_melt, name="total_melt"), xr.DataArray(runoff_rate, name="runoff_rate"), \
                             xr.DataArray(inst_smb, name="smb")])
    #glacier_melt = glacier_melt.assign_coords(water_year = ds["water_year"])

    return glacier_melt

glacier_melt = calculate_glaciermelt(degreedays_ds) # output in mm

# glacier_melt.total_melt.mean()
# glacier_melt.runoff_rate.mean()

## HBV
"""
Compute the runoff from the catchment with the HBV model
Python Code from the LHMP and adjusted to our needs (github.com/hydrogo/LHMP -
Ayzel Georgy. (2016). LHMP: lumped hydrological modelling playground. Zenodo. doi: 10.5281/zenodo.59501)
For the HBV model, evapotranspiration values are needed. These are calculated with the formula by Oudin et al. (2005) 
in the unit mm / day.
"""
print("Running the HBV model")

def simulation(df, parameters_HBV):
    # 1. new temporary dataframe from input with daily values
    df_hbv = df.resample("D").agg({"T2": 'mean', "RRR": 'sum'})

    if evap_data_available == True:
        evap = evap.resample("D").agg({"PE": 'sum'})
        df_hbv = evap["PE"]

    Temp = df_hbv['T2']
    if temp_unit == True:
        Temp = Temp - 273.15
    else:
        Temp = Temp

    Prec = df_hbv['RRR']
    if prec_unit == False:
        Prec = Prec / prec_conversion
    else:
        Prec = Prec

    # Calculation of PE with Oudin et al. 2005
    solar_constant = (1376 * 1000000) / 86400  # from 1376 J/m2s to MJm2d
    extra_rad = 27.086217947590317
    latent_heat_flux = 2.45
    water_density = 1000
    if evap_data_available == False:
        df_hbv["PE"] = np.where((df_hbv["T2"] - 273.15) + 5 > 0, ((extra_rad/(water_density*latent_heat_flux))* \
                                                              ((df_hbv["T2"] - 273.15) +5)/100)*1000, 0)
        Evap = df_hbv["PE"]
    else:
        Evap = df_hbv["PE"]

    # 2. set the parameters for the HBV
    parBETA, parCET, parFC, parK0, parK1, parK2, parLP, parMAXBAS,\
    parPERC, parUZL, parPCORR, parTT, parCFMAX, parSFCF, parCFR, parCWH = parameters_HBV

    # 3. initialize boxes and initial conditions
    # snowpack box
    SNOWPACK = np.zeros(len(Prec))
    SNOWPACK[0] = 0.0001
    # meltwater box
    MELTWATER = np.zeros(len(Prec))
    MELTWATER[0] = 0.0001
    # soil moisture box
    SM = np.zeros(len(Prec))
    SM[0] = 0.0001
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

    # 4. meteorological forcing preprocessing
    # overall correction factor
    Prec = parPCORR * Prec
    # precipitation separation
    # if T < parTT: SNOW, else RAIN
    RAIN = np.where(Temp  > parTT, Prec, 0)
    SNOW = np.where(Temp <= parTT, Prec, 0)
    # snow correction factor
    SNOW = parSFCF * SNOW
    # evaporation correction
    # a. calculate long-term averages of daily temperature
    Temp_mean = np.array([Temp.loc[Temp.index.dayofyear == x].mean()\
                          for x in range(1, 367)])
    # b. correction of Evaporation daily values
    Evap = Evap.index.map(lambda x: (1+parCET*(Temp[x] - Temp_mean[x.dayofyear - 1]))*Evap[x])
    # c. control Evaporation
    Evap = np.where(Evap > 0, Evap, 0)


    # 5. The main cycle of calculations
    for t in range(1, len(Qsim)):

        # 5.1 Snow routine
        # how snowpack forms
        SNOWPACK[t] = SNOWPACK[t-1] + SNOW[t]
        # how snowpack melts
        # day-degree simple melting
        melt = parCFMAX * (Temp[t] - parTT)
        # control melting
        if melt<0: melt = 0
        melt = min(melt, SNOWPACK[t])
        # how meltwater box forms
        MELTWATER[t] = MELTWATER[t-1] + melt
        # snowpack after melting
        SNOWPACK[t] = SNOWPACK[t] - melt
        # refreezing accounting
        refreezing = parCFR * parCFMAX * (parTT - Temp[t])
        # control refreezing
        if refreezing < 0: refreezing = 0
        refreezing = min(refreezing, MELTWATER[t])
        # snowpack after refreezing
        SNOWPACK[t] = SNOWPACK[t] + refreezing
        # meltwater after refreezing
        MELTWATER[t] = MELTWATER[t] - refreezing
        # recharge to soil
        tosoil = MELTWATER[t] - (parCWH * SNOWPACK[t]);
        # control recharge to soil
        if tosoil < 0: tosoil = 0
        # meltwater after recharge to soil
        MELTWATER[t] = MELTWATER[t] - tosoil

        # 5.2 Soil and evaporation routine
        # soil wetness calculation
        soil_wetness = (SM[t-1] / parFC)**parBETA
        # control soil wetness (should be in [0, 1])
        if soil_wetness < 0: soil_wetness = 0
        if soil_wetness > 1: soil_wetness = 1
        # soil recharge
        recharge = (RAIN[t] + tosoil) * soil_wetness
        # soil moisture update
        SM[t] = SM[t-1] + RAIN[t] + tosoil - recharge
        # excess of water calculation
        excess = SM[t] - parFC
        # control excess
        if excess < 0: excess = 0
        # soil moisture update
        SM[t] = SM[t] - excess

        # evaporation accounting
        evapfactor = SM[t] / (parLP * parFC)
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
        SUZ[t] = SUZ[t-1] + recharge + excess
        # percolation control
        perc = min(SUZ[t], parPERC)
        # update upper groudwater box
        SUZ[t] = SUZ[t] - perc
        # runoff from the highest part of upper grondwater box (surface runoff)
        Q0 = parK0 * max(SUZ[t] - parUZL, 0)
        # update upper groudwater box
        SUZ[t] = SUZ[t] - Q0
        # runoff from the middle part of upper groundwater box
        Q1 = parK1 * SUZ[t]
        # update upper groudwater box
        SUZ[t] = SUZ[t] - Q1
        # calculate lower groundwater box
        SLZ[t] = SLZ[t-1] + perc
        # runoff from lower groundwater box
        Q2 = parK2 * SLZ[t]
        # update lower groundwater box
        SLZ[t] = SLZ[t] - Q2

        # Total runoff calculation
        Qsim[t] = Q0 + Q1 + Q2

    # 6. Scale effect accounting
    # delay and smoothing simulated hydrograph
    # (Beck et al.,2016) used triangular transformation based on moving window
    # here are my method with simple forward filter based on Butterworht filter design
    # calculate Numerator (b) and denominator (a) polynomials of the IIR filter
    parMAXBAS = int(parMAXBAS)
    b, a = ss.butter(parMAXBAS, 1/parMAXBAS)
    # implement forward filter
    Qsim_smoothed = ss.lfilter(b, a, Qsim)
    # control smoothed runoff
    Qsim_smoothed = np.where(Qsim_smoothed > 0, Qsim_smoothed, 0)

    Qsim = Qsim_smoothed
    hbv_results = pd.DataFrame({"Q_HBV": Qsim, "HBV_snowpack": SNOWPACK, "HBV_soil_moisture": SM, "HBV_AET": ETact, \
                                "HBV_upper_gw": SUZ,"HBV_lower_gw": SLZ}, index=df_hbv.index)
    hbv_results = pd.concat([df_hbv, hbv_results], axis=1)
    return hbv_results

output_hbv = simulation(df, parameters_HBV)
## output dataframe
output = pd.concat([output_hbv, obs], axis=1)

Q_DDM = glacier_melt["runoff_rate"].mean(dim=["lat", "lon"])
Q_DDM = pd.array(Q_DDM)
output["Q_DDM"] = Q_DDM
output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]

output_csv = output.copy()
output_csv = output_csv.fillna(0)
output_csv.to_csv(output_path + "model_output_" +str(time_start[:4])+"-"+str(time_end[:4]+".csv"))
print("Writing the output csv to disc")