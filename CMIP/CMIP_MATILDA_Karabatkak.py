from pathlib import Path; home = str(Path.home())
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA_slim import MATILDA
from MATILDA_slim import MATILDA_functions

## Data
cmip_data = "/data/scratch/tappeana/Work/CMIP6_mean_42.25-78.25_2000-01-01-2099-12-31.csv"
cmip_data = home + "/Seafile/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/CMIP6_mean_42.25-78.25_2000-01-01-2099-12-31.csv"
parameter_df = pd.read_csv(home + "/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/best_param_kysyl_20210511.csv")

input_csv = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/ERA5/20210313_42.25-78.25_kyzylsuu_awsq_1982_2019.csv"
#input_csv = "/data/projects/ebaca/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/ERA5/20210313_42.25-78.25_kyzylsuu_awsq_1982_2019.csv"
#parameter = pd.read_csv(home +"/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/best_param_sa_0,7607.csv")

output_path = "/data/scratch/tappeana/Work/"

## Trend
hist_period_start=2001; hist_period_end = 2020; period_start = 2021; period_end = 2100; period_length = 20
variables = ["temp", "prec"]
scenarios = ["26", "45", "85"]

cmip_df = pd.read_csv(cmip_data)

cmip_df = cmip_df.set_index("time")
cmip_df.index = pd.to_datetime(cmip_df.index)
cmip_df["year"] = cmip_df.index.year
cmip_df["month"] = cmip_df.index.month

factors_cmip = MATILDA_functions.cmip_factors(cmip_df, variables, hist_period_start, hist_period_end, period_start, period_end, period_length)

parameter_df = pd.read_csv(home + "/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/Karabatkak_Catchment/best_param_kysyl_20210511.csv")
## MATILDA preparation -  Initial run: Kysylsuu
df = pd.read_csv(input_csv)

parameter = MATILDA.MATILDA_parameter(df, set_up_start='2000-01-01 00:00:00', set_up_end='2000-12-31 23:00:00',
                                      sim_start='2001-01-01 00:00:00', sim_end='2020-11-01 23:00:00', freq="D",
                                      lat=42.25, area_cat=315.69,
                                      area_glac=32.54, ele_dat=2550, ele_glac=4000, ele_cat=3221, parameter_df=parameter_df)
df_preproc = MATILDA.MATILDA_preproc(df, parameter)
df_preproc = df_preproc.resample("D").agg({"T2":"mean", "RRR":"sum"}) # to prepare future dataframes

output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter)  # MATILDA model run + downscaling
output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)

#MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path) # save regular MATILDA run

##
df_preproc["month"] = df_preproc.index.month

cmip_dfs = {}
cmip_dfs["df_hist"] = df_preproc.copy()
cmip_dfs["df_hist"].name = "df_hist"

cmip_dfs = MATILDA_functions.MATILDA_cmip_dfs(cmip_dfs, df_preproc, factors_cmip, hist_period_end, period_end, period_length, scenarios, variables)


## MATILDA run with CMIP data
cmip_output = pd.DataFrame(index=df_preproc.index)
cmip_output = cmip_output[parameter.sim_start:parameter.sim_end]

output_dict ={}

for i in cmip_dfs.keys():
    output_MATILDA = MATILDA.MATILDA_submodules(cmip_dfs[i], parameter) # MATILDA model run + downscaling
    output = output_MATILDA[0]["Q_Total"]
    cmip_output[i] = output
    output = output_MATILDA[0]
    output_dict[i] = output

# for i in output_dict.keys():
#     output_dict[i].to_csv(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/CMIP6/CMIP_runs/MATILDA_CMIP_kashkator_" + str(i) + ".csv")
cmip_output.to_csv(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/CMIP6/MATILDA_CMIP_kyzylsuu.csv")

## Glacier runs
cmip_output_glacier = pd.DataFrame(index=df_preproc.index)
cmip_output_glacier = cmip_output_glacier[parameter.sim_start:parameter.sim_end]

list_cmip_60 = {k: v for k, v in cmip_dfs.items() if k.endswith('2060')}
for i in list_cmip_60.keys():
    parameter_glac = parameter.copy()
    parameter_glac.area_glac = parameter_glac.area_glac *0.5
    output_MATILDA = MATILDA.MATILDA_submodules(list_cmip_60[i], parameter_glac)  # MATILDA model run + downscaling
    output = output_MATILDA[0]["Q_Total"]
    cmip_output_glacier[i] = output

# 2060-2080
list_cmip_80 = {k: v for k, v in cmip_dfs.items() if k.endswith('2080')}

for i in list_cmip_80:
    parameter_glac = parameter.copy()
    parameter_glac.area_glac = parameter_glac.area_glac * 0.4
    output_MATILDA = MATILDA.MATILDA_submodules(list_cmip_80[i], parameter_glac)  # MATILDA model run + downscaling
    output = output_MATILDA[0]["Q_Total"]
    cmip_output_glacier[i] = output

# 2080-2100
list_cmip_2100 = {k: v for k, v in cmip_dfs.items() if k.endswith('2100')}

for i in list_cmip_2100:
    parameter_glac = parameter.copy()
    parameter_glac.area_glac = parameter_glac.area_glac * 0.3
    output_MATILDA = MATILDA.MATILDA_submodules(list_cmip_2100[i], parameter_glac)  # MATILDA model run + downscaling
    output = output_MATILDA[0]["Q_Total"]
    cmip_output_glacier[i] = output

cmip_output_glacier.to_csv(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/CMIP6/MATILDA_CMIP_glacier-melt_kyzylsuu.csv")

## Plot preprocessing
cmip_output = pd.read_csv("/home/ana/Desktop/Meeting/MATILDA_CMIP_kyzylsuu.csv")
cmip_output_glacier = pd.read_csv("/home/ana/Desktop/Meeting/MATILDA_CMIP_kyzylsuu_glacier.csv")

cmip_output = cmip_output.set_index("TIMESTAMP")
cmip_output.index = pd.to_datetime(cmip_output.index)
cmip_output_glacier = cmip_output_glacier.set_index("TIMESTAMP")
cmip_output_glacier.index = pd.to_datetime(cmip_output_glacier.index)

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
# fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharey=True, figsize=(10, 6))
# gs = gridspec.GridSpec(2, 2)
# ax1 = plt.subplot(gs[0, 0])
# ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
# ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2040"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
# ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2040"], c="#009E73", linewidth=1.2, label="RCP 4.5")
# ax1.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2040"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
# ax2 = plt.subplot(gs[0,1], sharey=ax1)
# ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
# ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_26_2060"], c="#D55E00", linewidth=1.2, label="RCP 2.6")
# ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2060"], c="#009E73", linewidth=1.2, label="RCP 4.5")
# ax2.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2060"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
# ax3 = plt.subplot(gs[1,0], sharey=ax1)
# ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
# ax3.plot(cmip_output_monthly_mean["month"], annual.index.to_pydatetime(), c="#D55E00", linewidth=1.2, label="RCP 2.6")
# ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_45_2080"], c="#009E73", linewidth=1.2, label="RCP 4.5")
# ax3.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean["df_85_2080"], c="#CC79A7", linewidth=1.2, label="RCP 8.5")
# ax4 = plt.subplot(gs[1,1], sharey=ax1)
# ax4.plot(cmip_output_monthly_mean["month"], cmip_output_monthly_mean['df_hist'], c="#0072B2", linewidth=1.2, label="Historical")
# ax4.plot(cmip_output_monthly_mean["month"], annual.index.to_pydatetime(), c="#D55E00", linewidth=1.2, label="RCP 2.6")
# ax4.plot(cmip_output_monthly_mean["month"], annual.index.to_pydatetime(), c="#009E73", linewidth=1.2, label="RCP 4.5")
# ax4.plot(cmip_output_monthly_mean["month"], annual.index.to_pydatetime(), c="#CC79A7", linewidth=1.2, label="RCP 8.5")
# #ax1.legend(), ax2.legend(), ax3.legend(), ax4.legend()
# ax1.xaxis.set_ticks(np.arange(2, 12, 2)), ax2.xaxis.set_ticks(np.arange(2, 12, 2)), ax3.xaxis.set_ticks(np.arange(2, 12, 2)); ax4.xaxis.set_ticks(np.arange(2, 12, 2))
# fig.subplots_adjust(top=0.9, left=0.1, right=0.9, bottom=0.12)
# ax3.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12), ncol=4)
# ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9), ax4.set_ylabel("[mm]", fontsize=9)
# ax1.set_title("Period of 2021 - 2040", fontsize=9)
# ax2.set_title("Period of 2041 - 2060", fontsize=9)
# ax3.set_title("Period of 2061 - 2080", fontsize=9)
# ax4.set_title("Period of 2081 - 2100", fontsize=9)
# plt.show()
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
#ax3.legend(loc='upper center', bbox_to_anchor=(1.5, -0.5),fancybox=False, shadow=False, ncol=2)
plt.tight_layout()
plt.suptitle("MATILDA simulation for Kyzylsuu with CMIP6 forcing data")
plt.show()
#plt.savefig("/home/ana/Desktop/Meeting/CMIP6_Kyzylsuu_barplot.png", dpi=700)
##
annual = cmip_output.copy()
annual["month"] = annual.index.month
annual["day"] = annual.index.day
annual = annual.groupby(["month", "day"]).mean()
annual["date"] = pd.date_range('2020-01-01', freq='D', periods=len(annual)).strftime('%Y-%m-%d')
annual = annual.set_index(annual["date"])
annual.index = pd.to_datetime(annual.index)

annual_glacier = cmip_output_glacier.copy()
annual_glacier["month"] = annual_glacier.index.month
annual_glacier["day"] = annual_glacier.index.day
annual_glacier = annual_glacier.groupby(["month", "day"]).mean()
annual_glacier["date"] = pd.date_range('2020-01-01', freq='D', periods=len(annual_glacier)).strftime('%Y-%m-%d')
annual_glacier = annual_glacier.set_index(annual_glacier["date"])
annual_glacier.index = pd.to_datetime(annual_glacier.index)

## plot annual cycle
labels = ["January", "March", "May", "July", "September", "November", "January"]

fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(6, figsize=(10, 6))
gs = gridspec.GridSpec(3, 2)
ax1 = plt.subplot(gs[0, 0])
ax1.plot(annual.index.to_pydatetime(), annual["df_hist"], color='#2b2d2f')
ax1.plot(annual.index.to_pydatetime(), annual["df_26_2060"], color='#88CCEE')
ax1.plot(annual.index.to_pydatetime(), annual["df_26_2080"], color='#DDCC77')
ax1.plot(annual.index.to_pydatetime(), annual["df_26_2100"], color='#CC6677')
ax1.set_title("No glacier loss", fontsize=10)
plt.ylabel('Runoff [mm]')
ax1.set_xticklabels(labels, size = 8)
ax1.text(0.02, 0.95, 'RCP 2.6', transform=ax1.transAxes, fontsize=8, verticalalignment='top')
ax2 = plt.subplot(gs[1, 0], sharey=ax1)
# Make the plot
ax2.plot(annual.index.to_pydatetime(), annual["df_hist"], color='#2b2d2f')
ax2.plot(annual.index.to_pydatetime(), annual["df_45_2060"], color='#88CCEE')
ax2.plot(annual.index.to_pydatetime(), annual["df_45_2080"], color='#DDCC77')
ax2.plot( annual.index.to_pydatetime(), annual["df_45_2100"], color='#CC6677')
plt.ylabel('Runoff [mm]')
ax2.text(0.02, 0.95, 'RCP 4.5', transform=ax2.transAxes, fontsize=8, verticalalignment='top')
ax2.set_xticklabels(labels, size = 8)
ax3 = plt.subplot(gs[2, 0], sharey=ax1)
# Make the plot
ax3.plot(annual.index.to_pydatetime(), annual["df_hist"], color='#2b2d2f', label='2001-2020 (Reference period)')
ax3.plot(annual.index.to_pydatetime(), annual["df_85_2060"], color='#88CCEE', label="2041 - 2060 (GAL 50%)")
ax3.plot(annual.index.to_pydatetime(), annual["df_85_2080"], color='#DDCC77', label="2061 - 2080 (GAL 60%)")
ax3.plot(annual.index.to_pydatetime(), annual["df_85_2100"], color='#CC6677', label="2081 - 2100 (GAL 70%)")
# Adding Xticks
ax3.set_xticklabels(labels, size = 8)
plt.ylabel('Runoff [mm]')
ax3.text(0.02, 0.95, 'RCP 8.5', transform=ax3.transAxes, fontsize=8, verticalalignment='top')
#Glacier
ax4 = plt.subplot(gs[0, 1], sharey=ax1)
# Make the plot
ax4.plot(annual.index.to_pydatetime(), annual["df_hist"], color='#2b2d2f')
ax4.plot(annual.index.to_pydatetime(), annual_glacier["df_26_2060"], color='#88CCEE')
ax4.plot(annual.index.to_pydatetime(), annual_glacier["df_26_2080"], color='#DDCC77')
ax4.plot(annual.index.to_pydatetime(), annual_glacier["df_26_2100"], color='#CC6677')
ax4.set_title("Including glacier loss",  fontsize=10)
ax4.text(0.02, 0.95, 'RCP 2.6', transform=ax4.transAxes, fontsize=8, verticalalignment='top')
ax4.set_xticklabels(labels, size = 8)
ax5 = plt.subplot(gs[1, 1], sharey=ax1)
# Make the plot
ax5.plot(annual.index.to_pydatetime(), annual["df_hist"], color='#2b2d2f')
ax5.plot(annual.index.to_pydatetime(), annual_glacier["df_45_2060"], color='#88CCEE')
ax5.plot(annual.index.to_pydatetime(), annual_glacier["df_45_2080"], color='#DDCC77')
ax5.plot(annual.index.to_pydatetime(), annual_glacier["df_45_2100"], color='#CC6677')
ax5.text(0.02, 0.95, 'RCP 4.5', transform=ax5.transAxes, fontsize=8, verticalalignment='top')
ax5.set_xticklabels(labels, size = 8)
ax6 = plt.subplot(gs[2, 1], sharey=ax1)
# Make the plot
ax6.plot(annual.index.to_pydatetime(), annual["df_hist"], color='#2b2d2f')
ax6.plot(annual.index.to_pydatetime(), annual_glacier["df_85_2060"], color='#88CCEE')
ax6.plot(annual.index.to_pydatetime(), annual_glacier["df_85_2080"], color='#DDCC77')
ax6.plot(annual.index.to_pydatetime(), annual_glacier["df_85_2100"], color='#CC6677')
ax6.set_xticklabels(labels,  size = 8)
ax6.text(0.02, 0.95, 'RCP 8.5', transform=ax6.transAxes, fontsize=8, verticalalignment='top')
#ax3.legend(loc='upper center', bbox_to_anchor=(1.5, -0.5),fancybox=False, shadow=False, ncol=2)
plt.tight_layout()
plt.suptitle("Mean annual cycle for Kyzylsuu with CMIP6 forcing data")
plt.show()
#plt.savefig("/home/ana/Desktop/Meeting/CMIP6_Kyzylsuu_mean_annual.png", dpi=700)

## test output
path = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/CMIP6/CMIP_runs/"
CMIP_output = {}

import glob
for files in glob.glob(path + "*.csv"):
    df = pd.read_csv(files)
    name = files[122:132]
    CMIP_output[name] = df

