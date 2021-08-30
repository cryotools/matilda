## Building the glacier profile from the GIS data
from pathlib import Path; home = str(Path.home())
import pandas as pd
import numpy as np

##
# the following file is from the zonal statistics function from QGIS which calculates the mean ice thickness per elevation band
gis_thickness = "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/GIS/Bash-Kaindy/thickness_mean_glabtop.csv"
catchment_area = 46.224
elezone_interval = 100
def round_elezones(x, base=100):
    return base * round(x/base)

##
glacier_profile = pd.read_csv(home + gis_thickness)
glacier_profile.rename(columns={'elev_max':'Elevation'}, inplace=True)
glacier_profile["WE"] = glacier_profile["_mean"]*0.908*1000
glacier_profile = glacier_profile.drop(columns=["ID", "elev_min", "_mean"])
glacier_profile["Area"] = glacier_profile["Area"]/catchment_area

glacier_profile["EleZone"] = round_elezones(glacier_profile["Elevation"], base=elezone_interval)
glacier_profile = glacier_profile.sort_values(by='Elevation',ascending=True).reset_index(drop=True)

##
glacier_profile.to_csv("/home/ana/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/bash-kaindy_glacier_profile_tricht.csv", index=False)


