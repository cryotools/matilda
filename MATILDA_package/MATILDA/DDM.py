"""
Degree Day Model to calculate the accumulation, snow and ice melt and runoff rate from the glaciers.
Model input rewritten and adjusted to our needs from the pypdd function (github.com/juseg/pypdd
- # Copyright (c) 2013--2018, Julien Seguinot <seguinot@vaw.baug.ethz.ch>)
"""
import xarray as xr
import numpy as np

# Function to calculate the positive degree days on the glacier area
def calculate_PDD(ds):
    # masking the dataset to only get the glacier area
    if isinstance(ds, xr.Dataset):
        mask = ds.MASK.values
        temp = xr.where(mask==1, ds["T2"], np.nan)
        temp = temp.mean(dim=["lat", "lon"])
        temp = xr.where(temp>=100, temp - 273.15, temp) # making sure the temperature is in Celsius
        temp_min = temp.resample(time="D").min(dim="time")
        temp_max = temp.resample(time="D").max(dim="time")
        temp_mean = temp.resample(time="D").mean(dim="time")
        prec = xr.where(mask == 1, ds["RRR"], np.nan)
        prec = prec.mean(dim=["lat", "lon"])
        prec = prec.resample(time="D").sum(dim="time")
        time = temp_mean["time"]
    else:
        temp = ds["T2"]
        if temp[1] >= 100: # making sure the temperature is in Celsius
            temp = temp - 273.15
        temp_min = temp.resample("D").min()
        temp_mean = temp.resample("D").mean()
        temp_max = temp.resample("D").max()
        prec = ds["RRR"].resample("D").sum()

    pdd_ds = xr.merge([xr.DataArray(temp_mean, name="temp_mean"), xr.DataArray(temp_min, name="temp_min"), \
                   xr.DataArray(temp_max, name="temp_max"), xr.DataArray(prec)])

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
    pdd_ds["pdd"] = xr.where(pdd_ds["temp_mean"] > 0, pdd_ds["temp_mean"], 0)

    return pdd_ds

# Calculation of melt and runoff: input from the pypdd model
def calculate_glaciermelt(ds, pdd_factor_snow=2.8, pdd_factor_ice=5.6, temp_snow=0, temp_rain=2, refreeze_snow=0, refreeze_ice=0):
    # Initial Parameters for the DDM
    """
        *pdd_factor_snow* : float
            Positive degree-day factor for snow.
        *pdd_factor_ice* : float
            Positive degree-day factor for ice.
        *refreeze_snow* : float
            Refreezing fraction of melted snow.
        *refreeze_ice* : float
            Refreezing fraction of melted ice.
        *temp_snow* : float
            Temperature at which all precipitation falls as snow.
        *temp_rain* : float
            Temperature at which all precipitation falls as rain.

        Initial parameters
        'pdd_factor_snow': 2.8, # according to Huintjes et al. 2010  [5.7 mm per day per Celsius according to Hock 2003]
        'pdd_factor_ice': 5.6,  # according to Huintjes et al. 2010 [7.4 mm per day per Celsius according to Hock 2003]
        'temp_snow': 0.0,
        'temp_rain': 2.0,
        'refreeze_snow': 0.05,
        'refreeze_ice': 0.05}
    """

    temp = ds["temp_mean"]
    prec = ds["RRR"]
    pdd = ds["pdd"]

    """ pypdd.py line 311
        Compute accumulation rate from temperature and precipitation.
        The fraction of precipitation that falls as snow decreases linearly
        from one to zero between temperature thresholds defined by the
        `temp_snow` and `temp_rain` attributes.
    """
    reduced_temp = (temp_rain - temp) / (temp_rain - temp_snow)
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
        pot_snow_melt = pdd_factor_snow * pdd
    # effective snow melt can't exceed amount of snow
        snow_melt = np.minimum(snow, pot_snow_melt)
    # ice melt is proportional to excess snow melt
        ice_melt = (pot_snow_melt - snow_melt) * pdd_factor_ice / pdd_factor_snow
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
    runoff_rate = total_melt - refreeze_snow * snow_melt_rate \
                  - refreeze_ice * ice_melt_rate
    inst_smb = accu_rate - runoff_rate

    glacier_melt = xr.merge([xr.DataArray(inst_smb, name="DDM_smb"), xr.DataArray(accu_rate, name="DDM_accumulation_rate"), \
                            xr.DataArray(ice_melt_rate, name="DDM_ice_melt_rate"), xr.DataArray(snow_melt_rate, name="DDM_snow_melt_rate"), \
                            xr.DataArray(total_melt, name="DDM_total_melt"), xr.DataArray(runoff_rate, name="Q_DDM")])
    #glacier_melt = glacier_melt.assign_coords(water_year = ds["water_year"])

    # making the final dataframe
    DDM_results = glacier_melt.to_dataframe()
    DDM_results = DDM_results.round(3)


    return DDM_results
