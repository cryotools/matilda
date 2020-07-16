import numpy as np

lat = np.deg2rad(43)
# Part 2. Extraterrrestrial radiation calculation
# set solar constant (in W m-2)
Rsc = 1367
# calculate day of the year array
doy = np.array([i for i in range(1, 367)])
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
Re = Rsc * 86400 / np.pi * dr * (ws * np.sin(lat) * np.sin(dt) \
                                 + np.sin(ws) * np.cos(lat) * np.cos(dt))
# convert from J m-2 day-1 to MJ m-2 day-1
Re = Re / 10 ** 6