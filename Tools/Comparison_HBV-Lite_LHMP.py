## packages and functions
from pathlib import Path; home = str(Path.home())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

## files
lhmp = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/best_cosipyrun_no1_2011-18/best_cosipyrun_no1_hbv-output.csv")
hbv_light_glac = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/HBV-Light/HBV-light_data/Glacier_No.1/Python/Glacier_Run/Results/Results.txt", sep='\t')
hbv_light_noglac = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/HBV-Light/HBV-light_data/Glacier_No.1/Python/Noglacier_Run/Results/Results.txt", sep='\t')

## comparison
lhmp["Date"] = pd.to_datetime(lhmp["Date"])
lhmp.set_index(lhmp["Date"], inplace=True)
lhmp.index = pd.to_datetime(lhmp.index)
lhmp = lhmp.iloc[1:]

hbv_light_glac['Date'] = hbv_light_glac['Date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
hbv_light_glac["Date"] = pd.to_datetime(hbv_light_glac["Date"])
hbv_light_noglac['Date'] = hbv_light_noglac['Date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
hbv_light_noglac["Date"] = pd.to_datetime(hbv_light_noglac["Date"])

## Plots: Runoff
plt.figure(figsize=(10,6))
ax = plt.axis()
plt.plot(lhmp.index.to_pydatetime(), lhmp['Qobs'], alpha=0.5, label="Observations")
plt.plot(lhmp.index.to_pydatetime(), lhmp['Qsim'], label="LHMP")
plt.plot(lhmp.index.to_pydatetime(), hbv_light_glac['Qsim'], label="HBV Lite Glacier")
plt.plot(lhmp.index.to_pydatetime(), hbv_light_noglac['Qsim'], alpha=0.8, label="HBV Lite No Glacier")
plt.legend()
plt.ylabel("Daily runoff [mm]")
plt.xlabel("Time")
plt.show()

## Plots: Parameters
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
ax1.plot(lhmp.index.to_pydatetime(), lhmp["T2"], "red")
ax2.bar(lhmp.index.to_pydatetime(), lhmp["RRR"], width=10)
ax3.plot(lhmp.index.to_pydatetime(), lhmp["PE"], "green")
plt.xlabel("Date", fontsize=9)
ax1.grid(linewidth=0.25), ax2.grid(linewidth=0.25), ax3.grid(linewidth=0.25)
ax1.set_title("Temperature", fontsize=9)
ax2.set_title("Precipitation", fontsize=9)
ax3.set_title("Evapotranspiration", fontsize=9)
ax1.set_ylabel("[°C]", fontsize=9)
ax2.set_ylabel("[mm]", fontsize=9)
ax3.set_ylabel("[mm]", fontsize=9)
fig.suptitle("Meteorological input parameters", size=14)
plt.show()

fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(10,6))
ax1.plot(lhmp.index.to_pydatetime(), lhmp["AET"], "red", label="LHMP")
ax1.plot(lhmp.index.to_pydatetime(), hbv_light_noglac["AET"], "black", label="HBV Light")
ax2.plot(lhmp.index.to_pydatetime(), lhmp["soil_moisture"], "red", label="LHMP")
ax2.plot(lhmp.index.to_pydatetime(), hbv_light_noglac["SM"], "black", label="HBV Light")
ax3.plot(lhmp.index.to_pydatetime(), lhmp["snowpack"], "red", label="LHMP")
ax3.plot(lhmp.index.to_pydatetime(), hbv_light_noglac["Snow"], "black", label="HBV Light")
ax4.plot(lhmp.index.to_pydatetime(), lhmp["upper_gw"], "red", label="LHMP")
ax4.plot(lhmp.index.to_pydatetime(), hbv_light_noglac["SUZ"], "black", label="HBV Light")
ax5.plot(lhmp.index.to_pydatetime(), lhmp["lower_gw"], "red", label="LHMP")
ax5.plot(lhmp.index.to_pydatetime(), hbv_light_noglac["SLZ"], "black", label="HBV Light")
ax1.set_title("Actual evapotranspiration", fontsize=9)
ax2.set_title("Soil moisture", fontsize=9)
ax3.set_title("Water in snowpack", fontsize=9)
ax4.set_title("Upper groundwater box", fontsize=9)
ax5.set_title("Lower groundwater box", fontsize=9)
plt.xlabel("Date", fontsize=9)
plt.legend(loc='upper center', bbox_to_anchor=(0.8, -0.21),
          fancybox=True, shadow=True, ncol=5)
plt.show()
