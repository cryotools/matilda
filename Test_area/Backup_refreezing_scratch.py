import numpy as np
import xarray as xr

temp = ds["temp_mean"]
prec = ds["RRR"]
pdd = ds["pdd"]

reduced_temp = (parameter.TT_rain - temp) / (parameter.TT_rain - parameter.TT_snow)
snowfrac = np.clip(reduced_temp, 0, 1)
accu_rate = snowfrac * prec

# initialize snow depth and melt rates (pypdd.py line 214)
snow_depth = xr.zeros_like(temp)
snow_melt_rate = xr.zeros_like(temp)
ice_melt_rate = xr.zeros_like(temp)

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

##
# Ice fraction (Dichte_snow/Dichte_ice [kg/m³]):
rho_snow_fresh = 50     # Snow immediately after falling, in below-freezing temperatures, with no wind; fresh, uncompacted snow that has a high volume of trapped air
rho_snow_compact = 380  # Hard Wind Slab. Compacted snow after prolonged and heavy wind exposure.
rho_snow_firn = 600     # Advanced firn snow
rho_snow_slush = 720    # Advanced melting snow; snow/water mix
rho_ice = 917

theta_i = rho_snow / rho_ice

# Irreducible water content (CFR_snow_max):
theta_i = 0.785
if theta_i <= 0.23:
    theta_e = 0.0264 + 0.0099 * ((1-theta_i)/theta_i)
elif 0.23 < theta_i <= 0.812:
    theta_e = 0.08 - 0.1023 * (theta_i - 0.03)
else:
    theta_e = 0

# Cold content (melt temperature - mean temperature, MJ m^−2):
T_melt = 0              # melting temperature of ice
c_i = 2.1 * 10**-3      # specific heat of ice
cc_temp = T_melt - temp
cc_temp = xr.where(temp < 0, -temp, 0)      # cc_temp = T_melt - temp, T_melt = 0

cc = cc_temp * (snow_depth/1000) * c_i * rho_snow

# Refreezing factor (CFR_snow):

# theta_i between 0.055 (fresh snow) and 0.785 (slush) --> theta_e between 0.1965 and 0.0028 --> CFR_snow_max!
# --> Unit? Fraction?

# function of cc ?! How much is the volumetric content ratio changed by refreezing?
# --> COSIPY...




