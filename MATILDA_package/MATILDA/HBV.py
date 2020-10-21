"""
Compute the runoff from the catchment with the HBV model
Python Code from the LHMP and adjusted to our needs (github.com/hydrogo/LHMP -
Ayzel Georgy. (2016). LHMP: lumped hydrological modelling playground. Zenodo. doi: 10.5281/zenodo.59501)
For the HBV model, evapotranspiration values are needed. These are calculated with the formula by Oudin et al. (2005)
in the unit mm / day.

List of 16 HBV model parameters
    [parBETA, parCET,  parFC,    parK0,
    parK1,    parK2,   parLP,    parMAXBAS,
    parPERC,  parUZL,  parPCORR, parTT,
    parCFMAX, parSFCF, parCFR,   parCWH]

    # 16 PARAMETERS_HBV
    # BETA   - Parameter that determines the relative contribution to runoff from rain or snow melt
    #          [1, 6]
    # CET    - Evaporation correction factor
    #          (should be 0 if we don't want to change (Oudin et al., 2005) formula values)
    #          [0, 0.3]
    # FC     - Maximum soil moisture storage
    #          [50, 500]
    # K0     - Recession coefficient for surface soil box (upper part of SUZ)
    #          [0.01, 0.4]
    # K1     - Recession coefficient for upper groundwater box (main part of SUZ)
    #          [0.01, 0.4]
    # K2     - Recession coefficient for lower groundwater box (whole SLZ)
    #          [0.001, 0.15]
    # LP     - Threshold for reduction of evaporation (SM/FC)
    #          [0.3, 1]
    # MAXBAS - Routing parameter, order of Butterworth filter
    #          [1, 7]
    # PERC   - Percolation from soil to upper groundwater box
    #          [0, 3]
    # UZL    - Threshold parameter for groundwater boxes runoff (mm)
    #          [0, 500]
    # PCORR  - Precipitation (input sum) correction factor
    #          [0.5, 2]
    # TT     - Temperature which separates rain and snow fraction of precipitation
    #          [-1.5, 2.5]
    # CFMAX  - Snow melting rate (mm/day per Celsius degree)
    #          [1, 10]
    # SFCF   - Snowfall correction Factor
    #          [0.4, 1]
    # CFR    - Refreezing coefficient
    #          [0, 0.1] (usually 0.05)
    # CWH    - Fraction of meltwater and rainfall that retains in snowpack (water holding capacity)
    #          [0, 0.2] (usually 0.1)
"""

import pandas as pd
import scipy.signal as ss
import numpy as np
def hbv_simulation(df, cal_period_start, cal_period_end, parBETA=1.0, parCET=0.15,  parFC=250, parK0=0.055, parK1=0.055, \
                   parK2=0.04, parLP=0.7, parMAXBAS=3.0, parPERC=1.5, parUZL=120, parPCORR=1.0, parTT=0.0, parCFMAX=5.0, \
                   parSFCF=0.7, parCFR=0.05, parCWH=0.1):
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
    Temp_cal = Temp[cal_period_start:cal_period_end]
    Prec_cal = Prec[cal_period_start:cal_period_end]
    Evap_cal = Evap[cal_period_start:cal_period_end]
    # overall correction factor
    Prec_cal = parPCORR * Prec_cal
    # precipitation separation
    # if T < parTT: SNOW, else RAIN
    RAIN_cal = np.where(Temp_cal  > parTT, Prec_cal, 0)
    SNOW_cal = np.where(Temp_cal <= parTT, Prec_cal, 0)
    # snow correction factor
    SNOW_cal = parSFCF * SNOW_cal
    # evaporation correction
    # a. calculate long-term averages of daily temperature
    Temp_mean_cal = np.array([Temp_cal.loc[Temp_cal.index.dayofyear == x].mean()\
                          for x in range(1, 367)])
    # b. correction of Evaporation daily values
    Evap_cal = Evap_cal.index.map(lambda x: (1+parCET*(Temp_cal[x] - Temp_mean_cal[x.dayofyear - 1]))*Evap_cal[x])
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
        melt = parCFMAX * (Temp_cal[t] - parTT)
        # control melting
        if melt<0: melt = 0
        melt = min(melt, SNOWPACK_cal[t])
        # how meltwater box forms
        MELTWATER_cal[t] = MELTWATER_cal[t-1] + melt
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
        soil_wetness = (SM_cal[t-1] / parFC)**parBETA
        # control soil wetness (should be in [0, 1])
        if soil_wetness < 0: soil_wetness = 0
        if soil_wetness > 1: soil_wetness = 1
        # soil recharge
        recharge = (RAIN_cal[t] + tosoil) * soil_wetness
        # soil moisture update
        SM_cal[t] = SM_cal[t-1] + RAIN_cal[t] + tosoil - recharge
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
    print("HBV Spin up fished")

    # 3. meteorological forcing preprocessing for simulation
    # overall correction factor
    Prec = parPCORR * Prec
    # precipitation separation
    # if T < parTT: SNOW, else RAIN
    RAIN = np.where(Temp > parTT, Prec, 0)
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
    hbv_results = pd.DataFrame({"T2":Temp, "RRR":Prec, "PE":Evap, "HBV_snowpack": SNOWPACK, "HBV_soil_moisture": SM, "HBV_AET": ETact, \
                                "HBV_upper_gw": SUZ,"HBV_lower_gw": SLZ, "Q_HBV": Qsim}, index=df_hbv.index)
    hbv_results = hbv_results.round(3)
    return hbv_results
