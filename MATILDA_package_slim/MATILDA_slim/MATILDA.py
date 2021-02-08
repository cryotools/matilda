import xarray as xr
import numpy as np
import pandas as pd
import scipy.signal as ss

def MATILDA(df, obs, parameter):
    print('---')
    print('Starting the MATILDA simulation')
    # Downscaling of dataframe to mean catchment and glacier elevation
    def glacier_downscaling(df, parameter):
        height_diff_glacier = parameter.elevation_glacier - parameter.elevation_data
        height_diff_catchment = parameter.elevation_catchment - parameter.elevation_data

        df_glacier = df.copy()
        df_glacier["T2"] = np.where(df_glacier["T2"] <= 100, df_glacier["T2"] + 273.15, df_glacier["T2"])
        df_glacier["T2"] = df_glacier["T2"] + height_diff_glacier * float(parameter.lapse_rate_temperature)
        df_glacier["RRR"] = df_glacier["RRR"] + height_diff_glacier * float(parameter.lapse_rate_precipitation)

        df_catchment = df.copy()
        df_catchment["T2"] = np.where(df_catchment["T2"] <= 100, df_catchment["T2"] + 273.15, df_catchment["T2"])
        df_catchment["T2"] = df_catchment["T2"] + height_diff_catchment * float(parameter.lapse_rate_temperature)
        df_catchment["RRR"] = df_catchment["RRR"] + height_diff_catchment * float(
            parameter.lapse_rate_precipitation)
        return df_glacier, df_catchment

    df_glacier, df_catchment = glacier_downscaling(df, parameter)


    # Calculation of the positive degree days
    def calculate_PDD(ds):
        print("Calculating the positive degree days")
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

        # water_year = calc_hydrological_year(time)
        # pdd_ds = pdd_ds.assign_coords(water_year = water_year)

        # calculate the positive degree days
        pdd_ds["pdd"] = xr.where(pdd_ds["temp_mean"] > 0, pdd_ds["temp_mean"], 0)

        return pdd_ds

    degreedays_ds = calculate_PDD(df_glacier)

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
            [xr.DataArray(inst_smb, name="DDM_smb"), xr.DataArray(accu_rate, name="DDM_accumulation_rate"), \
             xr.DataArray(ice_melt_rate, name="DDM_ice_melt_rate"),
             xr.DataArray(snow_melt_rate, name="DDM_snow_melt_rate"), \
             xr.DataArray(total_melt, name="DDM_total_melt"), xr.DataArray(runoff_rate, name="Q_DDM")])
        # glacier_melt = glacier_melt.assign_coords(water_year = ds["water_year"])

        # making the final dataframe
        DDM_results = glacier_melt.to_dataframe()
        DDM_results = DDM_results.round(3)
        # scaling glacier melt to glacier area
        DDM_results["Q_DDM"] = DDM_results["Q_DDM"] * (parameter.glacier_area / parameter.catchment_area)
        print("Finished running the DDM")
        return DDM_results

    output_DDM = calculate_glaciermelt(degreedays_ds, parameter)

    """
    Compute the runoff from the catchment with the HBV model
    Python Code from the LHMP and adjusted to our needs (github.com/hydrogo/LHMP -
    Ayzel Georgy. (2016). LHMP: lumped hydrological modelling playground. Zenodo. doi: 10.5281/zenodo.59501)
    For the HBV model, evapotranspiration values are needed. These are calculated with the formula by Oudin et al. (2005)
    in the unit mm / day.
    """
    def hbv_simulation(df, parameter):
        print("Running the HBV model")
        # 1. new temporary dataframe from input with daily values
        if "PE" in df.columns:
            df_hbv = df.resample("D").agg({"T2": 'mean', "RRR": 'sum', "PE":"sum"})
        else:
           df_hbv = df.resample("D").agg({"T2": 'mean', "RRR": 'sum'})

        Temp = df_hbv['T2']
        if Temp[1] >= 100:  # making sure the temperature is in Celsius
            Temp = Temp - 273.15
        Prec = df_hbv['RRR']

        # Calculation of PE with Oudin et al. 2005
        solar_constant = (1376 * 1000000) / 86400  # from 1376 J/m2s to MJm2d
        extra_rad = 27.086217947590317
        latent_heat_flux = 2.45
        water_density = 1000
        if "PE" in df.columns:
            Evap = df_hbv["PE"]
        else:
            df_hbv["PE"] = np.where((Temp) + 5 > 0, ((extra_rad/(water_density*latent_heat_flux))* \
                                                                  ((Temp) +5)/100)*1000, 0)
            Evap = df_hbv["PE"]

        # 2. Calibration period:
        # 2.1 meteorological forcing preprocessing
        Temp_cal = Temp[parameter.cal_period_start:parameter.cal_period_end]
        Prec_cal = Prec[parameter.cal_period_start:parameter.cal_period_end]
        Evap_cal = Evap[parameter.cal_period_start:parameter.cal_period_end]
        # overall correction factor
        Prec_cal = parameter.PCORR * Prec_cal
        # precipitation separation
        # if T < parTT: SNOW, else RAIN
        RAIN_cal = np.where(Temp_cal > parameter.TT_rain, Prec_cal, 0)
        #SNOW_cal2 = np.where(Temp_cal <= parTT, Prec_cal, 0)
        reduced_temp_cal = (parameter.TT_rain - Temp_cal) / (parameter.TT_rain - parameter.TT_snow)
        snowfrac_cal = np.clip(reduced_temp_cal, 0, 1)
        SNOW_cal = snowfrac_cal * Prec_cal
        # snow correction factor
        SNOW_cal = parameter.SFCF * SNOW_cal
        # evaporation correction
        # a. calculate long-term averages of daily temperature
        Temp_mean_cal = np.array([Temp_cal.loc[Temp_cal.index.dayofyear == x].mean()\
                              for x in range(1, 367)])
        # b. correction of Evaporation daily values
        Evap_cal = Evap_cal.index.map(lambda x: (1+parameter.CET*(Temp_cal[x] - Temp_mean_cal[x.dayofyear - 1]))*Evap_cal[x])
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
            SNOWPACK_cal[t] = SNOWPACK_cal[t-1] + SNOW_cal[t]
            # how snowpack melts
            # day-degree simple melting
            melt = parameter.CFMAX_snow * (Temp_cal[t] - parameter.TT_snow)
            # control melting
            if melt<0: melt = 0
            melt = min(melt, SNOWPACK_cal[t])
            # how meltwater box forms
            MELTWATER_cal[t] = MELTWATER_cal[t-1] + melt
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
            soil_wetness = (SM_cal[t-1] / parameter.FC)**parameter.BETA
            # control soil wetness (should be in [0, 1])
            if soil_wetness < 0: soil_wetness = 0
            if soil_wetness > 1: soil_wetness = 1
            # soil recharge
            recharge = (RAIN_cal[t] + tosoil) * soil_wetness
            # soil moisture update
            SM_cal[t] = SM_cal[t-1] + RAIN_cal[t] + tosoil - recharge
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
        print("HBV Spin up fished")

        # 3. meteorological forcing preprocessing for simulation
        # overall correction factor
        Prec = parameter.PCORR * Prec
        # precipitation separation
        # if T < parTT: SNOW, else RAIN
        RAIN = np.where(Temp > parameter.TT_snow, Prec, 0)
        #SNOW = np.where(Temp <= parTT, Prec, 0)
        reduced_temp = (parameter.TT_rain - Temp) / (parameter.TT_rain - parameter.TT_snow)
        snowfrac = np.clip(reduced_temp, 0, 1)
        SNOW = snowfrac * Prec
        # snow correction factor
        SNOW = parameter.SFCF * SNOW
        # snow correction factor
        SNOW = parameter.SFCF * SNOW
        # evaporation correction
        # a. calculate long-term averages of daily temperature
        Temp_mean = np.array([Temp.loc[Temp.index.dayofyear == x].mean()\
                              for x in range(1, 367)])
        # b. correction of Evaporation daily values
        Evap = Evap.index.map(lambda x: (1+parameter.CET*(Temp[x] - Temp_mean[x.dayofyear - 1]))*Evap[x])
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
            SNOWPACK[t] = SNOWPACK[t-1] + SNOW[t]
            # how snowpack melts
            # day-degree simple melting
            melt = parameter.CFMAX_snow * (Temp[t] - parameter.TT_snow)
            # control melting
            if melt<0: melt = 0
            melt = min(melt, SNOWPACK[t])
            # how meltwater box forms
            MELTWATER[t] = MELTWATER[t-1] + melt
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
            soil_wetness = (SM[t-1] / parameter.FC)**parameter.BETA
            # control soil wetness (should be in [0, 1])
            if soil_wetness < 0: soil_wetness = 0
            if soil_wetness > 1: soil_wetness = 1
            # soil recharge
            recharge = (RAIN[t] + tosoil) * soil_wetness
            # soil moisture update
            SM[t] = SM[t-1] + RAIN[t] + tosoil - recharge
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
            SUZ[t] = SUZ[t-1] + recharge + excess
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
            SLZ[t] = SLZ[t-1] + perc
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
        b, a = ss.butter(parMAXBAS, 1/parMAXBAS)
        # implement forward filter
        Qsim_smoothed = ss.lfilter(b, a, Qsim)
        # control smoothed runoff
        Qsim_smoothed = np.where(Qsim_smoothed > 0, Qsim_smoothed, 0)

        Qsim = Qsim_smoothed
        hbv_results = pd.DataFrame({"T2":Temp, "RRR":Prec, "PE":Evap, "HBV_snowpack": SNOWPACK, "HBV_soil_moisture": SM, "HBV_AET": ETact, \
                                    "HBV_upper_gw": SUZ,"HBV_lower_gw": SLZ, "Q_HBV": Qsim}, index=df_hbv.index)
        hbv_results = hbv_results.round(3)
        print("Finished running the HBV")
        return hbv_results

    output_HBV = hbv_simulation(df_catchment, parameter)

    # Output postprocessing
    def output_postproc(output_HBV, output_DDM, obs):
        output = pd.concat([output_HBV, output_DDM], axis=1)
        output = pd.concat([output, obs], axis=1)
        output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]
        return output

    output_MATILDA = output_postproc(output_HBV, output_DDM, obs)
    output_MATILDA = output_MATILDA[parameter.sim_period_start:parameter.sim_period_end]

    # Nash–Sutcliffe model efficiency coefficient
    def NS(output_MATILDA, obs):
        nash_sut = 1 - np.sum((obs["Qobs"] - output_MATILDA["Q_Total"]) ** 2) / (np.sum((obs["Qobs"] - obs["Qobs"].mean()) ** 2))
        if nash_sut > 1 or nash_sut < -1:
            nash_sut = "error"
        return nash_sut

    nash_sut = NS(output_MATILDA, obs)

    if nash_sut == "error":
        print("ERROR. The Nash–Sutcliffe model efficiency coefficient is outside the range of -1 to 1")
    else:
        print("The Nash–Sutcliffe model efficiency coefficient of the MATILDA run is " + str(round(nash_sut, 2)))

    def create_statistics(output_MATILDA):
        stats = output_MATILDA.describe()
        sum = pd.DataFrame(output_MATILDA.sum())
        sum.columns = ["sum"]
        sum = sum.transpose()
        stats = stats.append(sum)
        stats = stats.round(3)
        return stats

    stats =  create_statistics(output_MATILDA)

    print(stats)
    print("End of the MATILDA simulation")
    print("---")

    output_all = [output_MATILDA, nash_sut, stats]
    return output_all

