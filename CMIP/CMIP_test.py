from datetime import datetime
from pathlib import Path; home = str(Path.home())
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA_slim import MATILDA

## Data
data_csv = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv"
cmip_trend = home + "/Seafile/Tianshan_data/CMIP/CMIP5/EC-EARTH_r6i1p1_r7i1p1_r8i1p1/CMIP5_monthly_trend.csv"

output = "test"

## Trend
cmip_df = pd.read_csv(home + "/Seafile/Tianshan_data/CMIP/CMIP5/EC-EARTH_r6i1p1_r7i1p1_r8i1p1/temp_prec_rcp26_rcp45_rcp85_2006-2100.csv")

cmip_df = cmip_df.set_index("time")
cmip_df.index = pd.to_datetime(cmip_df.index)
cmip_df["year"] = cmip_df.index.year
cmip_df["month"] = cmip_df.index.month

cmip_monthly = cmip_df.groupby(["month", "year"], as_index=False).agg(temp_26=('temp_26','mean'), temp_45=('temp_45','mean'),
                                                                      temp_85=('temp_85','mean'), prec_26= ('prec_26','sum'),
                                                                      prec_45= ('prec_45','sum'), prec_85= ('prec_85','sum'))

cmip_monthly["period"] = 0
cmip_monthly["period"] = np.where(((cmip_monthly["year"] >= 2006) & (cmip_monthly["year"] <= 2020)), "period_2006_2020", cmip_monthly["period"])
cmip_monthly["period"] = np.where(((cmip_monthly["year"] >= 2021) & (cmip_monthly["year"] <= 2040)), "period_2021_2040", cmip_monthly["period"])
cmip_monthly["period"] = np.where(((cmip_monthly["year"] >= 2041) & (cmip_monthly["year"] <= 2060)), "period_2041_2060", cmip_monthly["period"])
cmip_monthly["period"] = np.where(((cmip_monthly["year"] >= 2061) & (cmip_monthly["year"] <= 2080)), "period_2061_2080", cmip_monthly["period"])
cmip_monthly["period"] = np.where(((cmip_monthly["year"] >= 2081) & (cmip_monthly["year"] <= 2100)), "period_2081_2100", cmip_monthly["period"])

cmip_monthly_period = cmip_monthly.groupby(["month", "period"], as_index=False).agg(temp_26=('temp_26','mean'), temp_45=('temp_45','mean'),
                                                                      temp_85=('temp_85','mean'), prec_26= ('prec_26','mean'),
                                                                      prec_45= ('prec_45','mean'), prec_85= ('prec_85','mean'))

test = dict(tuple(cmip_monthly_period.groupby('period')))
test["period_2006_2020"]

