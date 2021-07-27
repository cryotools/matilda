## Running all the required functions
from pathlib import Path; home = str(Path.home())
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import GlabTop2

## Vergleich van der Tricht und Farinotti

farinotti = pd.read_csv("/home/ana/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/GIS/Kysylsuu/thickness_mean_karabatkak_farinotti.csv")
tricht = pd.read_csv("/home/ana/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/GIS/Kysylsuu/thickness_mean_karabatkak_tricht.csv")
glabtop = pd.read_csv("/home/ana/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/GIS/Kysylsuu/thickness_mean_karabatkak_glabtop.csv")

farinotti = farinotti.sort_values('elev_min')
tricht = tricht.sort_values('elev_min')
glabtop = glabtop.sort_values('elev_min')


plt.plot(farinotti["elev_min"], farinotti["_mean"], label="Farinotti")
plt.plot(tricht["elev_min"], tricht["_mean"], label="Tricht")
plt.plot(glabtop["elev_min"], glabtop["_mean"], label="Glabtop")
plt.legend()
plt.xlabel("Elevation [m]"), plt.ylabel("Thickness [m]")
plt.show()

## Vergleich Kashkator Output mit CMIP 4.5 run - gleiche Parameter + Input data

cmip_farinotti = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/output/new_deltaH/Kashkator/Kashkator_cmip_4_5_Farinotti_2021_2100_2021-07-27_11:07:46/glacier_area_2021-2100.csv")
cmip_tricht = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/output/new_deltaH/Kashkator/Kashkator_cmip_4_5_tricht_2021_2100_2021-07-27_13:10:41/glacier_area_2021-2100.csv")
cmip_glabtop = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/output/new_deltaH/Kashkator/Kashkator_cmip_4_5_glabtop_2021_2100_2021-07-27_11:31:51/glacier_area_2021-2100.csv")

cmip_comparison = cmip_farinotti[["time", "glacier_area"]].copy()
cmip_comparison = cmip_comparison.rename(columns = {'glacier_area':'area_farinotti'})
cmip_comparison["area_tricht"] = cmip_tricht["glacier_area"].copy()
cmip_comparison["area_glabtop"] = cmip_glabtop["glacier_area"].copy()

cmip_comparison['time'].iloc[0] = float(2020)
cmip_comparison['time'] = pd.to_numeric(cmip_comparison['time'], errors='coerce')
cmip_comparison["time"] = pd.to_datetime(cmip_comparison["time"], format='%Y')

plt.plot(cmip_comparison["time"], cmip_comparison["area_farinotti"], label="Farinotti")
plt.plot(cmip_comparison["time"], cmip_comparison["area_tricht"], label="Tricht")
plt.plot(cmip_comparison["time"], cmip_comparison["area_glabtop"], label="Glabtop")
plt.legend()
plt.xlabel(""), plt.ylabel("Glacier area [km2]")
plt.show()