import numpy as np
import xarray as xr
import pandas as pd
import salem
from pathlib import Path
import sys
import glob
home = str(Path.home())
working_directory = home + "/Desktop/in_cm4_8/"

cmip1 = working_directory + "temp/tas_day_INM-CM4-8_historical_r1i1p1f1_gr1_20130603-20141231.nc"
df_all2 = pd.DataFrame()

ds = xr.open_dataset(cmip1)
pick = ds.sel(lat=42.25, lon=78.25, method='nearest')
dat = pick["tas"].values
time = ds.indexes['time'].to_datetimeindex()
df = pd.DataFrame(dat, columns=["temp_hist"])
df = df.set_index(time)
df_all2 = df_all2.append(df)

df_all["temp_hist"] = df_all2["temp_hist"]

#df_all_all = df_all.copy()
df_all_all["prec_85"] = df_all["prec_85"]

df_all_all["prec_85"] = df_all_all["prec_85"] * 86400
df_all_all.to_csv(working_directory + "CMIP6_INM-CM4-8_20150101-21001230_42.75-78.0.csv")
df_all.to_csv(working_directory + "CMIP6_INM-CM4-8_19810917-20141231_42.75-78.0.csv")

df_test = df_all.copy()
df_test.drop('prec_hist', axis=1, inplace=True)
df_test.columns = ["temp_26"]
df_test["prec_85"] = df_all["prec_hist"]

df_all_all_all = df_test.append(df_all_all)
df_all_all_all.to_csv(working_directory + "CMIP6_INM-CM4-19810917-21001230_42.75-78.0.csv")