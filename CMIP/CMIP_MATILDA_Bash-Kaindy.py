from datetime import datetime
from pathlib import Path; home = str(Path.home())
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA_slim import MATILDA

## Data
input_csv = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/Old/no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv"
cmip_trend = home + "/Seafile/Tianshan_data/CMIP/CMIP5/EC-EARTH_r6i1p1_r7i1p1_r8i1p1/CMIP5_monthly_trend.csv"

output = "test"

## MATILDA preparation
df = pd.read_csv(input_csv)
cmip_trend = pd.read_csv(cmip_trend)

df = df.set_index("TIMESTAMP")
df.index = pd.to_datetime(df.index)
df = df["2001-01-01 00:00:00":"2018-11-01 23:00:00"]
df = df.resample("M").sum()
df.describe()

parameter = MATILDA.MATILDA_parameter(df, set_up_start='2000-01-01 00:00:00', set_up_end='2000-12-31 23:00:00',
                       sim_start='2001-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D", area_cat=46.224, area_glac=2.566,
                       ele_dat=3864, ele_glac=4035, ele_cat=3485)
df_preproc = MATILDA.MATILDA_preproc(df, parameter)
df_preproc = df_preproc.resample("D").agg({"T2":"mean", "RRR":"sum"})

## Preparing CMIP trend and dataframes for each period
cmip_trend_26_temp = cmip_trend[cmip_trend["scenario"] == "temp_26"]
cmip_trend_26_prec = cmip_trend[cmip_trend["scenario"] == "prec_26"]
cmip_trend_45_temp = cmip_trend[cmip_trend["scenario"] == "temp_45"]
cmip_trend_45_prec = cmip_trend[cmip_trend["scenario"] == "prec_45"]
cmip_trend_85_temp = cmip_trend[cmip_trend["scenario"] == "temp_85"]
cmip_trend_85_prec = cmip_trend[cmip_trend["scenario"] == "prec_85"]

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

#cmip_output.to_csv("/home/ana/Desktop/cmip_output_newroutine.csv")

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

cmip_output_glacier.to_csv("/home/ana/Desktop/cmip_output_glacier_newroutine.csv")

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
#ax3.legend(loc='upper center', bbox_to_anchor=(1.5, -0.5),fancybox=False, shadow=False, ncol=2)
plt.tight_layout()
plt.show()
plt.savefig("/home/ana/Desktop/CMIP_scenarios_runoff.png", dpi=700)
