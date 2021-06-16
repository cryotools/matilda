import pandas as pd
import numpy as np

## Factors for CMIP runs: Difference between historical run and specified time periods
# needs a dataframe with a timestamp as index and columns of temperature and precipitation data from different cmip scenarios
# these columns should start with "temp" and "prec"
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

# produces a dictionary with dataframes for each time period and scenario
# Uses the output from the cmip_factors function and the preprocessed dataframe

def MATILDA_cmip_dfs(cmip_dfs, df_preproc, factors_cmip, hist_period_end, period_end, period_length, scenarios, variables):
    for period in range(hist_period_end, period_end, period_length):
        for scen in scenarios:
            cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)] = df_preproc.copy()
            cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)].name = "df_" + str(scen) + "_" + str(period + period_length)
            for v in variables:
                factor = factors_cmip[v][factors_cmip[v]["scenario"].str.contains(scen)].copy()
                factor = factor.reset_index()
                if v == "temp":
                    for i in range(1, 13):
                        cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)]["T2"] = \
                            np.where(cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)]["month"] == i,
                                     cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)]["T2"] +
                                     factor.loc[i - 1, "temp_diff_hist_" + str(period + period_length)],
                                     cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)]["T2"])
                if v == "prec":
                    for i in range(1, 13):
                        cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)]["RRR"] = \
                            np.where(cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)]["month"] == i,
                                     cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)]["RRR"] *
                                     factor.loc[i - 1, "prec_fact_" + str(period + period_length)],
                                     cmip_dfs["df_" + str(scen) + "_" + str(period + period_length)]["RRR"])
    return cmip_dfs


