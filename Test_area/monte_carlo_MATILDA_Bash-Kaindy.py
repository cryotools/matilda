import random
import pandas as pd
import scipy.signal as ss
import numpy as np
import xarray as xr
from progress.bar import Bar
from MATILDA import dataformatting, DDM

df = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182ERA5_Land_2018_2019_down.csv")
obs = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/runoff_bashkaindy_2019.csv")
cal_period_start = '2019-01-01 00:00:00' # beginning of  period
cal_period_end = '2019-12-31 23:00:00' # end of period: one year is recommended
# Time period of the model simulation
sim_period_start = '2019-01-01 00:00:00' # beginning of simulation period
sim_period_end = '2019-12-31 23:00:00'
glacier_area = 2.566
catchment_area = 46.232
df = dataformatting.data_preproc(df, cal_period_start, sim_period_end) # formatting the input to right format
obs = dataformatting.data_preproc(obs, cal_period_start, sim_period_end)
ix = pd.date_range(start=sim_period_start, end=sim_period_end, freq='D')
obs = obs.reindex(ix)
df_DDM = dataformatting.glacier_downscaling(df, height_diff=682, lapse_rate_temperature=-0.006, lapse_rate_precipitation=0)

degreedays_ds = DDM.calculate_PDD(df_DDM)

# DDM parameter: minimum and maximum values
pdd_snow_min, pdd_snow_max = [1,8]
pdd_ice_min, pdd_ice_max = [1,8]
temp_snow_min, temp_snow_max = [-1.5, 2]
temp_rain_min, temp_rain_max = [0, 2]
refreeze_snow_min, refreeze_snow_max = [0, 0.1]
refreeze_ice_min, refreeze_ice_max = [0, 0.1]

# HBV parameter: minimum and maximum values
BETA_min, BETA_max = [1,6]
CET_min, CET_max = [0, 0.3]
FC_min, FC_max = [50, 500]
K0_min, K0_max = [0.01, 0.4]
K1_min, K1_max = [0.01, 0.4]
K2_min, K2_max = [0.001, 0.15]
LP_min, LP_max = [0.3, 1]
MAXBAS_min, MAXBAS_max = [2, 7]
PERC_min, PERC_max = [0, 3]
UZL_min, UZL_max = [0, 500]
PCORR_min, PCORR_max = [0.5, 2]
TT_min, TT_max = [-1.5, 2]
TT_rain_min, TT_rain_max = [0, 2]
CFMAX_min, CFMAX_max = [1, 10]
SFCF_min, SFCF_max = [0.4, 1]
CFR_min, CFR_max = [0, 0.1]
CWH_min, CWH_max = [0, 0.2]

# Monte Carlo Simulation for HBV
def monte_carlo(ds, df, obs, cal_period_start, cal_period_end, n):
    monte_carlo_results = []
    bar = Bar('Monte Carlo run', max=n)

    for i in range(n):
        bar.next()

        # get a random value for each DDM parameter
        pdd_factor_snow = random.uniform(pdd_snow_min, pdd_snow_max)
        pdd_factor_ice = random.uniform(pdd_ice_min, pdd_ice_max)
        temp_snow = random.uniform(temp_snow_min, temp_snow_max)
        if temp_snow > temp_rain_min:
            temp_rain_min == temp_snow
        temp_rain = random.uniform(temp_rain_min, temp_rain_max)
        refreeze_snow = random.uniform(refreeze_snow_min, refreeze_snow_max)
        refreeze_ice = random.uniform(refreeze_ice_min, refreeze_ice_max)

        # get a random value for each HBV parameter
        parBETA = random.randrange(BETA_min, BETA_max)
        parCET = random.uniform(CET_min, CET_max)
        parFC = random.randrange(FC_min, FC_max)
        parK0 = random.uniform(K0_min, K0_max)
        parK1 = random.uniform(K1_min, K1_max)
        parK2 = random.uniform(K2_min, K2_max)
        parLP = random.uniform(LP_min, LP_max)
        parMAXBAS = random.uniform(MAXBAS_min, MAXBAS_max)
        parPERC = random.uniform(PERC_min, PERC_max)
        parUZL = random.randrange(UZL_min, UZL_max)
        parPCORR = random.uniform(PCORR_min, PCORR_max)
        parTT = random.uniform(TT_min, TT_max)
        if parTT > TT_rain_min:
            TT_rain_min == parTT
        parTT_rain = random.uniform(TT_rain_min, TT_rain_max)
        parCFMAX = random.uniform(CFMAX_min, CFMAX_max)
        parSFCF = random.uniform(SFCF_min, SFCF_max)
        parCFR = random.uniform(CFR_min, CFR_max)
        parCWH = random.uniform(CWH_min, CWH_max)

        # actual DDM simulation
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
        runoff_rate = runoff_rate * (glacier_area / catchment_area)  # scaling glacier melt to glacier area
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


        # actual HBV simulation
        # 1. new temporary dataframe from input with daily values
        if "PE" in df.columns:
            df_hbv = df.resample("D").agg({"T2": 'mean', "RRR": 'sum', "PE": "sum"})
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
            df_hbv["PE"] = np.where((Temp) + 5 > 0, ((extra_rad / (water_density * latent_heat_flux)) * \
                                                     ((Temp) + 5) / 100) * 1000, 0)
            Evap = df_hbv["PE"]

        # 2. Calibration period:
        # 2.1 meteorological forcing preprocessing
        Temp_cal = Temp[cal_period_start:cal_period_end]
        Prec_cal = Prec[cal_period_start:cal_period_end]
        Evap_cal = Evap[cal_period_start:cal_period_end]
        # overall correction factor
        Prec_cal = parPCORR * Prec_cal
        # precipitation separation
        # if T < parTT: SNOW, else RAIN
        RAIN_cal = np.where(Temp_cal > parTT_rain, Prec_cal, 0)
        # SNOW_cal2 = np.where(Temp_cal <= parTT, Prec_cal, 0)
        reduced_temp_cal = (parTT_rain - Temp_cal) / (parTT_rain - parTT)
        snowfrac_cal = np.clip(reduced_temp_cal, 0, 1)
        SNOW_cal = snowfrac_cal * Prec_cal
        # snow correction factor
        SNOW_cal = parSFCF * SNOW_cal
        # evaporation correction
        # a. calculate long-term averages of daily temperature
        Temp_mean_cal = np.array([Temp_cal.loc[Temp_cal.index.dayofyear == x].mean() \
                                  for x in range(1, 367)])
        # b. correction of Evaporation daily values
        Evap_cal = Evap_cal.index.map(
            lambda x: (1 + parCET * (Temp_cal[x] - Temp_mean_cal[x.dayofyear - 1])) * Evap_cal[x])
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
            melt = parCFMAX * (Temp_cal[t] - parTT)
            # control melting
            if melt < 0: melt = 0
            melt = min(melt, SNOWPACK_cal[t])
            # how meltwater box forms
            MELTWATER_cal[t] = MELTWATER_cal[t - 1] + melt
            # snowpack after melting
            SNOWPACK_cal[t] = SNOWPACK_cal[t] - melt
            # refreezing accounting
            refreezing = parCFR * parCFMAX * (parTT - Temp_cal[t])
            # control refreezing
            if refreezing < 0: refreezing = 0
            refreezing = min(refreezing, MELTWATER_cal[t])
            # snowpack after refreezing
            SNOWPACK_cal[t] = SNOWPACK_cal[t] + refreezing
            # meltwater after refreezing
            MELTWATER_cal[t] = MELTWATER_cal[t] - refreezing
            # recharge to soil
            tosoil = MELTWATER_cal[t] - (parCWH * SNOWPACK_cal[t]);
            # control recharge to soil
            if tosoil < 0: tosoil = 0
            # meltwater after recharge to soil
            MELTWATER_cal[t] = MELTWATER_cal[t] - tosoil

            # 2.3.1 Soil and evaporation routine
            # soil wetness calculation
            soil_wetness = (SM_cal[t - 1] / parFC) ** parBETA
            # control soil wetness (should be in [0, 1])
            if soil_wetness < 0: soil_wetness = 0
            if soil_wetness > 1: soil_wetness = 1
            # soil recharge
            recharge = (RAIN_cal[t] + tosoil) * soil_wetness
            # soil moisture update
            SM_cal[t] = SM_cal[t - 1] + RAIN_cal[t] + tosoil - recharge
            # excess of water calculation
            excess = SM_cal[t] - parFC
            # control excess
            if excess < 0: excess = 0
            # soil moisture update
            SM_cal[t] = SM_cal[t] - excess

            # evaporation accounting
            evapfactor = SM_cal[t] / (parLP * parFC)
            # control evapfactor in range [0, 1]
            if evapfactor < 0: evapfactor = 0
            if evapfactor > 1: evapfactor = 1
            # calculate actual evaporation
            ETact_cal[t] = Evap_cal[t] * evapfactor
            # control actual evaporation
            ETact_cal[t] = min(SM_cal[t], ETact_cal[t])

            # last soil moisture updating
            SM_cal[t] = SM_cal[t] - ETact_cal[t]

        # 3. meteorological forcing preprocessing for simulation
        # overall correction factor
        Prec = parPCORR * Prec
        # precipitation separation
        # if T < parTT: SNOW, else RAIN
        RAIN = np.where(Temp > parTT, Prec, 0)
        # SNOW = np.where(Temp <= parTT, Prec, 0)
        reduced_temp = (parTT_rain - Temp) / (parTT_rain - parTT)
        snowfrac = np.clip(reduced_temp, 0, 1)
        SNOW = snowfrac * Prec
        # snow correction factor
        SNOW = parSFCF * SNOW
        # snow correction factor
        SNOW = parSFCF * SNOW
        # evaporation correction
        # a. calculate long-term averages of daily temperature
        Temp_mean = np.array([Temp.loc[Temp.index.dayofyear == x].mean() \
                              for x in range(1, 367)])
        # b. correction of Evaporation daily values
        Evap = Evap.index.map(lambda x: (1 + parCET * (Temp[x] - Temp_mean[x.dayofyear - 1])) * Evap[x])
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
            melt = parCFMAX * (Temp[t] - parTT)
            # control melting
            if melt < 0: melt = 0
            melt = min(melt, SNOWPACK[t])
            # how meltwater box forms
            MELTWATER[t] = MELTWATER[t - 1] + melt
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
            soil_wetness = (SM[t - 1] / parFC) ** parBETA
            # control soil wetness (should be in [0, 1])
            if soil_wetness < 0: soil_wetness = 0
            if soil_wetness > 1: soil_wetness = 1
            # soil recharge
            recharge = (RAIN[t] + tosoil) * soil_wetness
            # soil moisture update
            SM[t] = SM[t - 1] + RAIN[t] + tosoil - recharge
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
            SUZ[t] = SUZ[t - 1] + recharge + excess
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
            SLZ[t] = SLZ[t - 1] + perc
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
        b, a = ss.butter(parMAXBAS, 1 / parMAXBAS)
        # implement forward filter
        Qsim_smoothed = ss.lfilter(b, a, Qsim)
        # control smoothed runoff
        Qsim_smoothed = np.where(Qsim_smoothed > 0, Qsim_smoothed, 0)

        Qsim = Qsim_smoothed

        Q_Total = Qsim + runoff_rate

        nash_sut = 1 - np.sum((obs["Qobs"] - Q_Total) ** 2) / (np.sum((obs["Qobs"] - obs["Qobs"].mean()) ** 2))

        monte_carlo_results.append({"Run":i, "BETA":parBETA, "CET":parCET, "FC":parFC, "K0":parK0, "K1":parK1, "K2":parK2, "LP":parLP, \
                "MAXBAS":parMAXBAS, "PERC":parPERC, "UZL":parUZL, "PCORR":parPCORR, "TT":parTT, "CFMAX": parCFMAX, \
                "SFCF":parSFCF, "CFR":parCFR, "CWH":parCWH, "PDD_snow":pdd_factor_snow, "PDD_ice":pdd_factor_ice, \
                "DDM_temp_snow":temp_snow, "DDM_temp_rain":temp_rain, "refreeze_snow":refreeze_snow, "refreeze_ice":refreeze_ice, "Nash Sutcliff":nash_sut})
    monte_carlo_results = pd.DataFrame(monte_carlo_results)
    monte_carlo_results = monte_carlo_results.round(3)
    bar.finish()

    return monte_carlo_results

monte_carlo_results = monte_carlo(degreedays_ds, df, obs, cal_period_start, cal_period_end, 1000)
monte_carlo_results.to_csv("/home/ana/Seafile/SHK/Scripts/centralasiawaterresources/Test_area/monte_carlo_results_bash-kaindy.csv")
results_100 = pd.read_csv("/home/ana/Seafile/SHK/Scripts/centralasiawaterresources/Test_area/monte_carlo_results_bash-kaindy.csv")
results_100 = results_100.sort_values(by=['Nash Sutcliff'], ascending=False)

