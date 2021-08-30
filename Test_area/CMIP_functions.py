## Calculate factors to determine the mean change in temperature and precipitation of specified time periods of CMIP data in comparison to the historical ERA5 run
from datetime import datetime
from pathlib import Path; home = str(Path.home())
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA_slim import MATILDA

## Data
# CMIP timeseries
cmip_data = home + "/Seafile/Tianshan_data/CMIP/CMIP6/all_models/CMIP6_mean_42.25-78.25_2000-01-01-2099-12-31.csv"
#ERA5 data to calculate the factor
input_csv = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/ERA5/20210313_42.25-78.25_kyzylsuu_awsq_1982_2019.csv"

hist_period_start=2001; hist_period_end = 2020; period_start = 2021; period_end = 2099; period_length = 20
variables = ["temp", "prec"]

##
cmip_df = pd.read_csv(cmip_data)

cmip_df = cmip_df.set_index("time")
cmip_df.index = pd.to_datetime(cmip_df.index)
cmip_df["year"] = cmip_df.index.year
cmip_df["month"] = cmip_df.index.month
cmip_df["period"] = 0


def cmip_factors(cmip_df, variables, hist_period_start, hist_period_end, period_start, period_end, period_length):
    cmip_df["year"] = cmip_df.index.year
    cmip_df["month"] = cmip_df.index.month
    cmip_df["period"] = 0

    for i in range(hist_period_start, hist_period_end):
        cmip_df["period"] = np.where((cmip_df["year"] >= i) &
                                            (cmip_df["year"] <= i + (period_length - 1)),
                                           "hist_period", cmip_df["period"])


    for i in range(period_start, period_end, period_length):
        cmip_df["period"] = np.where((cmip_df["year"] >= i) &
                                            (cmip_df["year"] <= i + (period_length - 1)),
                                           "period_" + str(i) + "_" + str(i + (period_length - 1)), cmip_df["period"])


    cmip_df_temp = cmip_df.loc[:, ~cmip_df.columns.str.startswith('pr')]
    cmip_df_prec = cmip_df.loc[:, ~cmip_df.columns.str.startswith('temp')]
    cmip_df_temp = cmip_df_temp.groupby(["month", "year", "period"], as_index=False).mean()
    cmip_df_prec = cmip_df_prec.groupby(["month", "year", "period"], as_index=False).sum()
    cmip_monthly = pd.merge(cmip_df_temp, cmip_df_prec)
    cmip_monthly = cmip_monthly.drop(columns="year")


    monthly_trend_cmip = cmip_monthly.melt(id_vars=['month', 'period'])
    monthly_trend_cmip = monthly_trend_cmip.pivot_table(index=['month','variable'], columns='period',values='value')
    monthly_trend_cmip = monthly_trend_cmip.reset_index()
    monthly_trend_cmip.rename(columns={'variable':'scenario'}, inplace=True)


    for i in range(period_start, period_end, period_length):
        monthly_trend_cmip["temp_diff_hist_"+str(i+(period_length-1))] = monthly_trend_cmip["period_" + str(i) + "_" + str(i + (period_length - 1))] - monthly_trend_cmip["hist_period"]
        monthly_trend_cmip["prec_fact_"+str(i+(period_length-1))] = monthly_trend_cmip["period_" + str(i) + "_" + str(i + (period_length - 1))] / monthly_trend_cmip["hist_period"]

    scenario = np.unique(monthly_trend_cmip["scenario"])

    factors = {}
    for i in variables:
        all_factors = monthly_trend_cmip.loc[monthly_trend_cmip["scenario"].str.startswith(i)].copy()
        factors[i] = all_factors.loc[:, (all_factors.columns.str.startswith(("month", "scenario", i)))]

    return factors

factors_cmip = cmip_factors(cmip_df, variables, hist_period_start, hist_period_end, period_start, period_end, period_length)


