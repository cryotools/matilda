## packages and functions
from pathlib import Path; home = str(Path.home())
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

## files
lhmp = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/LHMP/output.csv")
hbv_light_glac = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/HBV_Light/HBV-light_data/Glacier_No.1/Python/Glacier_Run/Results/Results.txt", sep='\t')
hbv_light_noglac = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/HBV_Light/HBV-light_data/Glacier_No.1/Python/Noglacier_Run/Results/Results.txt", sep='\t')

## comparison
lhmp["Date"] = pd.to_datetime(lhmp["Date"])
lhmp = lhmp.rename(columns={"Qsim":"Qsim_lhmp"})
hbv_light_glac = hbv_light_glac.rename(columns={"Qsim":"Qsim_hbv_light_glac"})
hbv_light_noglac = hbv_light_noglac.rename(columns={"Qsim":"Qsim_hbv_light_noglac"})

hbv_light_glac['Date'] = hbv_light_glac['Date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
hbv_light_glac["Date"] = pd.to_datetime(hbv_light_glac["Date"])
hbv_light_noglac['Date'] = hbv_light_noglac['Date'].apply(lambda x: datetime.strptime(str(x),'%Y%m%d'))
hbv_light_noglac["Date"] = pd.to_datetime(hbv_light_noglac["Date"])

comparison = pd.merge(lhmp, hbv_light_glac, on="Date")
comparison = pd.merge(comparison, hbv_light_noglac, on="Date")
comparison = comparison[["Date", "Qobs_x", "Qsim_lhmp", "Qsim_hbv_light_glac", "Qsim_hbv_light_noglac"]]

comparison["diff_noglac"] = abs(comparison["Qsim_lhmp"] - comparison["Qsim_hbv_light_noglac"])
## Plots
plt.plot(comparison["Date"], comparison["diff_noglac"], label="Difference")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
ax = plt.axis()
plt.plot(comparison["Date"], comparison['Qobs_x'], alpha=0.5, label="Observations")
plt.plot(comparison["Date"], comparison['Qsim_lhmp'], label="LHMP")
plt.plot(comparison["Date"], comparison['Qsim_hbv_light_glac'], label="HBV Lite Glacier")
plt.plot(comparison["Date"], comparison['Qsim_hbv_light_noglac'], alpha=0.8, label="HBV Lite No Glacier")
plt.legend()
plt.ylabel("Daily runoff [mm]")
plt.xlabel("Time")
plt.show()
