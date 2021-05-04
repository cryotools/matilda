from datetime import datetime
from pathlib import Path; home = str(Path.home())
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA_slim import MATILDA

## Data
cmip_data = home + "/Seafile/Tianshan_data/CMIP/CMIP6/in_cm4_8/CMIP6_INM-CM4-19810917-21001230_42.75-78.0.csv"

input_csv = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/ERA5/20210313_42.25-78.25_kyzylsuu_awsq_1982_2019.csv"
parameter = pd.read_csv(home +"/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/best_param_sa_0,7607.csv")

output_path = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/output/Kyzyluu_CMIP6/"

## Trend
cmip_df = pd.read_csv(cmip_data)

cmip_df = cmip_df.set_index("Unnamed: 0")
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

monthly_trend_cmip = cmip_monthly_period.melt(id_vars=['month', 'period'])
monthly_trend_cmip = monthly_trend_cmip.pivot_table(index=['month','variable'], columns='period',values='value')
monthly_trend_cmip = monthly_trend_cmip.reset_index()
monthly_trend_cmip.rename(columns={'variable':'scenario'}, inplace=True)


monthly_trend_cmip["diff_hist_2040"] = monthly_trend_cmip["period_2021_2040"] - monthly_trend_cmip["period_2006_2020"]
monthly_trend_cmip["diff_hist_2060"] = monthly_trend_cmip["period_2041_2060"] - monthly_trend_cmip["period_2006_2020"]
monthly_trend_cmip["diff_hist_2080"] = monthly_trend_cmip["period_2061_2080"] - monthly_trend_cmip["period_2006_2020"]
monthly_trend_cmip["diff_hist_2100"] = monthly_trend_cmip["period_2081_2100"] - monthly_trend_cmip["period_2006_2020"]
monthly_trend_cmip["prec_fact_2040"] = monthly_trend_cmip["period_2021_2040"] / monthly_trend_cmip["period_2006_2020"]
monthly_trend_cmip["prec_fact_2060"] = monthly_trend_cmip["period_2041_2060"] / monthly_trend_cmip["period_2006_2020"]
monthly_trend_cmip["prec_fact_2080"] = monthly_trend_cmip["period_2061_2080"] / monthly_trend_cmip["period_2006_2020"]
monthly_trend_cmip["prec_fact_2100"] = monthly_trend_cmip["period_2081_2100"] / monthly_trend_cmip["period_2006_2020"]

## MATILDA preparation
df = pd.read_csv(input_csv)

parameter = MATILDA.MATILDA_parameter(df, set_up_start='2000-01-01 00:00:00', set_up_end='2000-12-31 23:00:00',
                       sim_start='2001-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D", area_cat=315.69,
              area_glac=31.51, ele_dat=2550, ele_glac=4000, ele_cat=3221, lr_temp=-0.005,
                                  lr_prec=-0.000048, TT_snow=0.616, \
                                  TT_rain=0.679, CFMAX_snow=7.812, CFMAX_ice=8.89, CFR_snow=0.06198, \
                                  CFR_ice=0.0847, BETA=6, CET=0.01508, FC=151.6, K0=0.2734, K1=0.0566, K2=0.0765, LP=0.9595, \
                                  MAXBAS=3.016, PERC=1.266, UZL=498, PCORR=1.324, SFCF=0.703, CWH=0.0000003)
df_preproc = MATILDA.MATILDA_preproc(df, parameter)
df_preproc = df_preproc.resample("D").agg({"T2":"mean", "RRR":"sum"}) # to prepare future dataframes

output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter)  # MATILDA model run + downscaling
output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)

#MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path) # save regular MATILDA run

## Preparing CMIP trend and dataframes for each period
cmip_trend_26_temp = monthly_trend_cmip[monthly_trend_cmip["scenario"] == "temp_26"]
cmip_trend_26_prec = monthly_trend_cmip[monthly_trend_cmip["scenario"] == "prec_26"]
cmip_trend_45_temp = monthly_trend_cmip[monthly_trend_cmip["scenario"] == "temp_45"]
cmip_trend_45_prec = monthly_trend_cmip[monthly_trend_cmip["scenario"] == "prec_45"]
cmip_trend_85_temp = monthly_trend_cmip[monthly_trend_cmip["scenario"] == "temp_85"]
cmip_trend_85_prec = monthly_trend_cmip[monthly_trend_cmip["scenario"] == "prec_85"]

cmip_trend_26_temp = cmip_trend_26_temp.reset_index()
cmip_trend_26_prec = cmip_trend_26_prec.reset_index()
cmip_trend_45_temp = cmip_trend_45_temp.reset_index()
cmip_trend_45_prec = cmip_trend_45_prec.reset_index()
cmip_trend_85_temp = cmip_trend_85_temp.reset_index()
cmip_trend_85_prec = cmip_trend_85_prec.reset_index()

df_hist = df_preproc.copy()
df_hist.name = "df_hist"

df_preproc["month"] = df_preproc.index.month

df_26_2040 = df_preproc.copy()
df_26_2040.name = "df_26_2040"
df_26_2060 = df_preproc.copy()
df_26_2060.name = "df_26_2060"
df_26_2080 = df_preproc.copy()
df_26_2080.name = "df_26_2080"
df_26_2100 = df_preproc.copy()
df_26_2100.name = "df_26_2100"
df_45_2040 = df_preproc.copy()
df_45_2040.name = "df_45_2040"
df_45_2060 = df_preproc.copy()
df_45_2060.name = "df_45_2060"
df_45_2080 = df_preproc.copy()
df_45_2080.name = "df_45_2080"
df_45_2100 = df_preproc.copy()
df_45_2100.name = "df_45_2100"
df_85_2040 = df_preproc.copy()
df_85_2040.name = "df_85_2040"
df_85_2060 = df_preproc.copy()
df_85_2060.name = "df_85_2060"
df_85_2080 = df_preproc.copy()
df_85_2080.name = "df_85_2080"
df_85_2100 = df_preproc.copy()
df_85_2100.name = "df_85_2100"

for i in range(1, 13):
    df_26_2040["T2"] = np.where(df_26_2040["month"] == i, df_26_2040["T2"] + cmip_trend_26_temp.loc[i-1, "diff_hist_2040"], df_26_2040["T2"])
    df_26_2040["RRR"] = np.where(df_26_2040["month"] == i, df_26_2040["RRR"] * cmip_trend_26_prec.loc[i - 1, "prec_fact_2040"], df_26_2040["RRR"])
for i in range(1, 13):
    df_26_2060["T2"] = np.where(df_26_2060["month"] == i, df_26_2060["T2"] + cmip_trend_26_temp.loc[i-1, "diff_hist_2060"], df_26_2060["T2"])
    df_26_2060["RRR"] = np.where(df_26_2060["month"] == i, df_26_2060["RRR"] * cmip_trend_26_prec.loc[i - 1, "prec_fact_2060"], df_26_2060["RRR"])
for i in range(1, 13):
    df_26_2080["T2"] = np.where(df_26_2080["month"] == i, df_26_2080["T2"] + cmip_trend_26_temp.loc[i-1, "diff_hist_2080"], df_26_2080["T2"])
    df_26_2080["RRR"] = np.where(df_26_2080["month"] == i, df_26_2080["RRR"] * cmip_trend_26_prec.loc[i - 1, "prec_fact_2080"], df_26_2080["RRR"])
for i in range(1, 13):
    df_26_2100["T2"] = np.where(df_26_2100["month"] == i, df_26_2100["T2"] + cmip_trend_26_temp.loc[i-1, "diff_hist_2100"], df_26_2100["T2"])
    df_26_2100["RRR"] = np.where(df_26_2100["month"] == i, df_26_2100["RRR"] * cmip_trend_26_prec.loc[i - 1, "prec_fact_2100"], df_26_2100["RRR"])
for i in range(1, 13):
    df_45_2040["T2"] = np.where(df_45_2040["month"] == i, df_45_2040["T2"] + cmip_trend_45_temp.loc[i-1, "diff_hist_2040"], df_45_2040["T2"])
    df_45_2040["RRR"] = np.where(df_45_2040["month"] == i, df_45_2040["RRR"] * cmip_trend_45_prec.loc[i - 1, "prec_fact_2040"], df_45_2040["RRR"])
for i in range(1, 13):
    df_45_2060["T2"] = np.where(df_45_2060["month"] == i, df_45_2060["T2"] + cmip_trend_45_temp.loc[i-1, "diff_hist_2060"], df_45_2060["T2"])
    df_45_2060["RRR"] = np.where(df_45_2060["month"] == i, df_45_2060["RRR"] * cmip_trend_45_prec.loc[i - 1, "prec_fact_2060"], df_45_2060["RRR"])
for i in range(1, 13):
    df_45_2080["T2"] = np.where(df_45_2080["month"] == i, df_45_2080["T2"] + cmip_trend_45_temp.loc[i-1, "diff_hist_2080"], df_45_2080["T2"])
    df_45_2080["RRR"] = np.where(df_45_2080["month"] == i, df_45_2080["RRR"] * cmip_trend_45_prec.loc[i - 1, "prec_fact_2080"], df_45_2080["RRR"])
for i in range(1, 13):
    df_45_2100["T2"] = np.where(df_45_2100["month"] == i, df_45_2100["T2"] + cmip_trend_45_temp.loc[i-1, "diff_hist_2100"], df_45_2100["T2"])
    df_45_2100["RRR"] = np.where(df_45_2100["month"] == i, df_45_2100["RRR"] * cmip_trend_45_prec.loc[i - 1, "prec_fact_2100"], df_45_2100["RRR"])
for i in range(1, 13):
    df_85_2040["T2"] = np.where(df_85_2040["month"] == i, df_85_2040["T2"] + cmip_trend_85_temp.loc[i-1, "diff_hist_2040"], df_85_2040["T2"])
    df_85_2040["RRR"] = np.where(df_85_2040["month"] == i, df_85_2040["RRR"] * cmip_trend_85_prec.loc[i - 1, "prec_fact_2040"], df_85_2040["RRR"])
for i in range(1, 13):
    df_85_2060["T2"] = np.where(df_85_2060["month"] == i, df_85_2060["T2"] + cmip_trend_85_temp.loc[i-1, "diff_hist_2060"], df_85_2060["T2"])
    df_85_2060["RRR"] = np.where(df_85_2060["month"] == i, df_85_2060["RRR"] * cmip_trend_85_prec.loc[i - 1, "prec_fact_2060"], df_85_2060["RRR"])
for i in range(1, 13):
    df_85_2080["T2"] = np.where(df_85_2080["month"] == i, df_85_2080["T2"] + cmip_trend_85_temp.loc[i-1, "diff_hist_2080"], df_85_2080["T2"])
    df_85_2080["RRR"] = np.where(df_85_2080["month"] == i, df_85_2080["RRR"] * cmip_trend_85_prec.loc[i - 1, "prec_fact_2080"], df_85_2080["RRR"])
for i in range(1, 13):
    df_85_2100["T2"] = np.where(df_85_2100["month"] == i, df_85_2100["T2"] + cmip_trend_85_temp.loc[i-1, "diff_hist_2100"], df_85_2100["T2"])
    df_85_2100["RRR"] = np.where(df_85_2100["month"] == i, df_85_2100["RRR"] * cmip_trend_85_prec.loc[i - 1, "prec_fact_2100"], df_85_2100["RRR"])

## MATILDA run with CMIP data
cmip_output = pd.DataFrame(index=df_preproc.index)
cmip_output = cmip_output[parameter.sim_start:parameter.sim_end]

list_cmip = [df_hist, df_26_2040, df_26_2060, df_26_2080, df_26_2100, df_45_2040, df_45_2060, df_45_2080, df_45_2100, df_85_2040, df_85_2060, df_85_2080, df_85_2100]


for i in list_cmip:
    output_MATILDA = MATILDA.MATILDA_submodules(i, parameter) # MATILDA model run + downscaling
    output = output_MATILDA[0]["Q_Total"]
    cmip_output[i.name] = output

#cmip_output.to_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/output/Kyzyluu_CMIP6/MATILDA_CMIP.csv")

## Glacier runs

list_cmip_60 = [df_26_2060, df_45_2060, df_85_2060]
cmip_output_glacier = pd.DataFrame(index=df_preproc.index)
cmip_output_glacier = cmip_output_glacier[parameter.sim_start:parameter.sim_end]


for i in list_cmip_60:
    parameter_glac = parameter.copy()
    parameter_glac.area_glac = parameter_glac.area_glac *0.5
    output_MATILDA = MATILDA.MATILDA_submodules(i, parameter_glac)  # MATILDA model run + downscaling
    output = output_MATILDA[0]["Q_Total"]
    cmip_output_glacier[i.name] = output

# 2060-2080
list_cmip_80 = [df_26_2080, df_45_2080, df_85_2080]

for i in list_cmip_80:
    parameter_glac = parameter.copy()
    parameter_glac.area_glac = parameter_glac.area_glac *0.4
    output_MATILDA = MATILDA.MATILDA_submodules(i, parameter_glac)  # MATILDA model run + downscaling
    output = output_MATILDA[0]["Q_Total"]
    cmip_output_glacier[i.name] = output

# 2080-2100
list_cmip_2100 = [df_26_2100, df_45_2100, df_85_2100]

for i in list_cmip_2100:
    parameter_glac = parameter.copy()
    parameter_glac.area_glac = parameter_glac.area_glac * 0.3
    output_MATILDA = MATILDA.MATILDA_submodules(i, parameter_glac)  # MATILDA model run + downscaling
    output = output_MATILDA[0]["Q_Total"]
    cmip_output_glacier[i.name] = output

#cmip_output_glacier.to_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/output/Kyzyluu_CMIP6/MATILDA_CMIP_glacier-melt.csv")

## Plot preprocessing
cmip_output_monthly = cmip_output.resample("M").agg("sum")
cmip_output_monthly["month"] = cmip_output_monthly.index.month
cmip_output_monthly_mean = cmip_output_monthly.groupby(["month"]).mean()
cmip_output_monthly_mean["month"] = cmip_output_monthly_mean.index
cmip_output_monthly_mean["month2"] = cmip_output_monthly_mean["month"] - 0.25

cmip_output_glacier_monthly = cmip_output_glacier.resample("M").agg("sum")
cmip_output_glacier_monthly["month"] = cmip_output_glacier_monthly.index.month
cmip_output_glacier_monthly_mean = cmip_output_glacier_monthly.groupby(["month"]).mean()
cmip_output_glacier_monthly_mean["month"] = cmip_output_glacier_monthly_mean.index

## Plots
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharey=True, figsize=(10, 6))
gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0, 0])
ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2040"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2040"], c="#009E73", linewidth=1.2, label="RCP 4.5")
ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2040"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
ax2 = plt.subplot(gs[0,1], sharey=ax1)
ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2060"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2060"], c="#009E73", linewidth=1.2, label="RCP 4.5")
ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2060"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
ax3 = plt.subplot(gs[1,0], sharey=ax1)
ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2080"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2080"], c="#009E73", linewidth=1.2, label="RCP 4.5")
ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2080"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
ax4 = plt.subplot(gs[1,1], sharey=ax1)
ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2100"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2100"], c="#009E73", linewidth=1.2, label="RCP 4.5")
ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2100"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
#ax1.legend(), ax2.legend(), ax3.legend(), ax4.legend()
ax1.xaxis.set_ticks(np.arange(2, 12, 2)), ax2.xaxis.set_ticks(np.arange(2, 12, 2)), ax3.xaxis.set_ticks(np.arange(2, 12, 2)); ax4.xaxis.set_ticks(np.arange(2, 12, 2))
fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9), ax4.set_ylabel("[mm]", fontsize=9)
ax1.set_title("Period of 2021 - 2040", fontsize=9)
ax2.set_title("Period of 2041 - 2060", fontsize=9)
ax3.set_title("Period of 2061 - 2080", fontsize=9)
ax4.set_title("Period of 2081 - 2100", fontsize=9)
plt.show()
##
barWidth = 0.2
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(10, 6))
gs = gridspec.GridSpec(3, 2)
ax1 = plt.subplot(gs[0, 0])
# Set position of bar on X axis
br1 = np.arange(len(cmip_output_monthly_mean["df_26_2060"]))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
# Make the plot
ax1.bar(br1, cmip_output_monthly_mean["df_hist"], color='#2b2d2f', width=barWidth, edgecolor='grey')
ax1.bar(br2, cmip_output_monthly_mean["df_26_2060"], color='#88CCEE', width=barWidth, edgecolor='grey')
ax1.bar(br3, cmip_output_monthly_mean["df_26_2080"], color='#DDCC77', width=barWidth, edgecolor='grey')
ax1.bar(br4, cmip_output_monthly_mean["df_26_2100"], color='#CC6677', width=barWidth, edgecolor='grey')
#mystep(cmip_output_monthly_mean["month2"], cmip_output_monthly_mean["df_hist"], ax=ax1, color="k", linewidth=0.5)
ax1.set_title("No glacier loss", fontsize=10)
plt.ylabel('Runoff [mm]')
ax1.text(0.02, 0.95, 'RCP 2.6', transform=ax1.transAxes, fontsize=8, verticalalignment='top')
plt.xticks([r + barWidth for r in range(len(cmip_output_monthly_mean["month"]))], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
# Adding Xticks
ax2 = plt.subplot(gs[1, 0], sharey=ax1)
# Make the plot
ax2.bar(br1, cmip_output_monthly_mean["df_hist"], color='#2b2d2f', width=barWidth,edgecolor='grey')
ax2.bar(br2, cmip_output_monthly_mean["df_45_2060"], color='#88CCEE', width=barWidth, edgecolor='grey')
ax2.bar(br3, cmip_output_monthly_mean["df_45_2080"], color='#DDCC77', width=barWidth, edgecolor='grey')
ax2.bar(br4, cmip_output_monthly_mean["df_45_2100"], color='#CC6677', width=barWidth, edgecolor='grey')
plt.ylabel('Runoff [mm]')
ax2.text(0.02, 0.95, 'RCP 4.5', transform=ax2.transAxes, fontsize=8, verticalalignment='top')
plt.xticks([r + barWidth for r in range(len(cmip_output_monthly_mean["month"]))], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
ax3 = plt.subplot(gs[2, 0], sharey=ax1)
# Make the plot
ax3.bar(br1, cmip_output_monthly_mean["df_hist"], color='#2b2d2f', width=barWidth,edgecolor='grey', label='2001-2020 (Reference period)')
ax3.bar(br2, cmip_output_monthly_mean["df_85_2060"], color='#88CCEE', width=barWidth,
        edgecolor='grey', label="2041 - 2060 (GAL 50%)")
ax3.bar(br3, cmip_output_monthly_mean["df_85_2080"], color='#DDCC77', width=barWidth,
        edgecolor='grey', label="2061 - 2080 (GAL 60%)")
ax3.bar(br4, cmip_output_monthly_mean["df_85_2100"], color='#CC6677', width=barWidth,
        edgecolor='grey', label="2081 - 2100 (GAL 70%)")
# Adding Xticks
plt.xlabel('Month')
plt.ylabel('Runoff [mm]')
ax3.text(0.02, 0.95, 'RCP 8.5', transform=ax3.transAxes, fontsize=8, verticalalignment='top')
plt.xticks([r + barWidth for r in range(len(cmip_output_monthly_mean["month"]))], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
#Glacier
ax4 = plt.subplot(gs[0, 1], sharey=ax1)
# Make the plot
ax4.bar(br1, cmip_output_monthly_mean["df_hist"], color='#2b2d2f', width=barWidth,edgecolor='grey')
ax4.bar(br2, cmip_output_glacier_monthly_mean["df_26_2060"], color='#88CCEE', width=barWidth, edgecolor='grey')
ax4.bar(br3, cmip_output_glacier_monthly_mean["df_26_2080"], color='#DDCC77', width=barWidth, edgecolor='grey')
ax4.bar(br4, cmip_output_glacier_monthly_mean["df_26_2100"], color='#CC6677', width=barWidth, edgecolor='grey')
ax4.set_title("Including glacier loss",  fontsize=10)
ax4.text(0.02, 0.95, 'RCP 2.6', transform=ax4.transAxes, fontsize=8, verticalalignment='top')
plt.xticks([r + barWidth for r in range(len(cmip_output_monthly_mean["month"]))], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
ax5 = plt.subplot(gs[1, 1], sharey=ax1)
# Make the plot
ax5.bar(br1, cmip_output_monthly_mean["df_hist"], color='#2b2d2f', width=barWidth,edgecolor='grey')
ax5.bar(br2, cmip_output_glacier_monthly_mean["df_45_2060"], color='#88CCEE', width=barWidth, edgecolor='grey')
ax5.bar(br3, cmip_output_glacier_monthly_mean["df_45_2080"], color='#DDCC77', width=barWidth, edgecolor='grey')
ax5.bar(br4, cmip_output_glacier_monthly_mean["df_45_2100"], color='#CC6677', width=barWidth, edgecolor='grey')
plt.xticks([r + barWidth for r in range(len(cmip_output_monthly_mean["month"]))], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
ax5.text(0.02, 0.95, 'RCP 4.5', transform=ax5.transAxes, fontsize=8, verticalalignment='top')
ax6 = plt.subplot(gs[2, 1], sharey=ax1)
# Make the plot
ax6.bar(br1, cmip_output_monthly_mean["df_hist"], color='#2b2d2f', width=barWidth,edgecolor='grey')
ax6.bar(br2, cmip_output_glacier_monthly_mean["df_85_2060"], color='#88CCEE', width=barWidth, edgecolor='grey')
ax6.bar(br3, cmip_output_glacier_monthly_mean["df_85_2080"], color='#DDCC77', width=barWidth, edgecolor='grey')
ax6.bar(br4, cmip_output_glacier_monthly_mean["df_85_2100"], color='#CC6677', width=barWidth, edgecolor='grey')
plt.xlabel('Month')
ax6.text(0.02, 0.95, 'RCP 8.5', transform=ax6.transAxes, fontsize=8, verticalalignment='top')
plt.xticks([r + barWidth for r in range(len(cmip_output_monthly_mean["month"]))], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
ax3.legend(loc='upper center', bbox_to_anchor=(1.5, -0.5),fancybox=False, shadow=False, ncol=2)
plt.tight_layout()
plt.suptitle("MATILDA simulation for the Kyzyluu catchment with CMIP6 forcing data")
plt.show()
#plt.savefig("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/output/Kyzyluu_CMIP6/output_barplot2.png", dpi=700)