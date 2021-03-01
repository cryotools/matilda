import numpy as np
import scipy.signal as ss

def hbv_simulation(Temp, Prec, Evap, BETA, CET, FC, K0, K1, K2, LP, MAXBAS, PERC, UZL, PCORR, TT_snow, TT_rain, CFMAX_snow, SFCF, CFR_snow, CWH):
    print("Running the HBV model")
    # 3. meteorological forcing preprocessing for simulation
    # overall correction factor
    Prec = PCORR * Prec
    # precipitation separation
    # if T < parTT: SNOW, else RAIN
    RAIN = np.where(Temp > TT_snow, Prec, 0)
    # SNOW = np.where(Temp <= parTT, Prec, 0)
    reduced_temp = (TT_rain - Temp) / (TT_rain - TT_snow)
    snowfrac = np.clip(reduced_temp, 0, 1)
    SNOW = snowfrac * Prec
    # snow correction factor
    SNOW = SFCF * SNOW
    # snow correction factor
    SNOW = SFCF * SNOW
    # evaporation correction
    # a. calculate long-term averages of daily temperature
    Temp_mean = np.array([Temp.loc[Temp.index.dayofyear == x].mean() \
                          for x in range(1, 367)])
    # b. correction of Evaporation daily values
    Evap = Evap.index.map(lambda x: (1 + CET * (Temp[x] - Temp_mean[x.dayofyear - 1])) * Evap[x])
    # c. control Evaporation
    Evap = np.where(Evap > 0, Evap, 0)

    # 4. initialize boxes and initial conditions after calibration
    # snowpack box
    SNOWPACK = np.zeros(len(Prec))
    # meltwater box
    MELTWATER = np.zeros(len(Prec))
    MELTWATER[0] = 0.0001
    # soil moisture box
    SM = np.zeros(len(Prec))
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
        melt = CFMAX_snow * (Temp[t] - TT_snow)
        # control melting
        if melt < 0: melt = 0
        melt = min(melt, SNOWPACK[t])
        # how meltwater box forms
        MELTWATER[t] = MELTWATER[t - 1] + melt
        # snowpack after melting
        SNOWPACK[t] = SNOWPACK[t] - melt
        # refreezing accounting
        refreezing = CFR_snow * CFMAX_snow * (TT_snow - Temp[t])
        # control refreezing
        if refreezing < 0: refreezing = 0
        refreezing = min(refreezing, MELTWATER[t])
        # snowpack after refreezing
        SNOWPACK[t] = SNOWPACK[t] + refreezing
        # meltwater after refreezing
        MELTWATER[t] = MELTWATER[t] - refreezing
        # recharge to soil
        tosoil = MELTWATER[t] - (CWH * SNOWPACK[t]);
        # control recharge to soil
        if tosoil < 0: tosoil = 0
        # meltwater after recharge to soil
        MELTWATER[t] = MELTWATER[t] - tosoil

        # 5.2 Soil and evaporation routine
        # soil wetness calculation
        soil_wetness = (SM[t - 1] / FC) ** BETA
        # control soil wetness (should be in [0, 1])
        if soil_wetness < 0: soil_wetness = 0
        if soil_wetness > 1: soil_wetness = 1
        # soil recharge
        recharge = (RAIN[t] + tosoil) * soil_wetness
        # soil moisture update
        SM[t] = SM[t - 1] + RAIN[t] + tosoil - recharge
        # excess of water calculation
        excess = SM[t] - FC
        # control excess
        if excess < 0: excess = 0
        # soil moisture update
        SM[t] = SM[t] - excess

        # evaporation accounting
        evapfactor = SM[t] / (LP * FC)
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
        perc = min(SUZ[t], PERC)
        # update upper groudwater box
        SUZ[t] = SUZ[t] - perc
        # runoff from the highest part of upper grondwater box (surface runoff)
        Q0 = K0 * max(SUZ[t] - UZL, 0)
        # update upper groudwater box
        SUZ[t] = SUZ[t] - Q0
        # runoff from the middle part of upper groundwater box
        Q1 = K1 * SUZ[t]
        # update upper groudwater box
        SUZ[t] = SUZ[t] - Q1
        # calculate lower groundwater box
        SLZ[t] = SLZ[t - 1] + perc
        # runoff from lower groundwater box
        Q2 = K2 * SLZ[t]
        # update lower groundwater box
        SLZ[t] = SLZ[t] - Q2

        # Total runoff calculation
        Qsim[t] = Q0 + Q1 + Q2

    # 6. Scale effect accounting
    # delay and smoothing simulated hydrograph
    # (Beck et al.,2016) used triangular transformation based on moving window
    # here are my method with simple forward filter based on Butterworht filter design
    # calculate Numerator (b) and denominator (a) polynomials of the IIR filter
    parMAXBAS = int(MAXBAS)
    b, a = ss.butter(parMAXBAS, 1 / parMAXBAS)
    # implement forward filter
    Qsim_smoothed = ss.lfilter(b, a, Qsim)
    # control smoothed runoff
    Qsim_smoothed = np.where(Qsim_smoothed > 0, Qsim_smoothed, 0)

    Qsim = Qsim_smoothed
    return Qsim