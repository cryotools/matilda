# -*- coding: UTF-8 -*-
"""
MATILDA (Modeling wATer resources In gLacierizeD cAtchments) is a combination of a degree day model and the HBV model (Bergstöm 1976) to compute total runoff of glacierized catchments.
This file may use the input files created by the COSIPY-utility "aws2cosipy" as forcing data and or a simple dataframe with temperature, precipitation and if possible evapotranspiration and additional observation runoff data to validate it.
"""
## import of necessary packages
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from MATILDA_slim import MATILDA

## Setting file paths and parameters
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/"
input_path_data = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/best_cosipyrun_no1/"
input_path_observations = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"

data_csv = "best_cosipy_input_no1_2000-20.csv" # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)

output_path = working_directory + "input_output/output/" + data_csv[:15]


df = pd.read_csv(input_path_data + data_csv)

## Running MATILDA
parameter = MATILDA.MATILDA_parameter(df, set_up_start='2000-01-01 00:00:00', set_up_end='2001-12-31 23:00:00',
                       sim_start='2001-01-01 00:00:00', sim_end='2099-11-01 23:00:00', freq="Y", area_cat=3.367, area_glac=1.581,
                       ele_dat=4025, ele_glac=4036, ele_cat=4025)
df = MATILDA.MATILDA_preproc(df, parameter) # Data preprocessing

# # building a dataframe for the future
# df_future = df.copy()
# for i in range(4):
#     df_add = df.copy()
#     df_add.index = df_add.index + pd.offsets.DateOffset(years=((i+1)*20))
#     df_add["T2"] = df_add["T2"] + ((i+1) * 0.5)
#     df_add["period"] = (i+1) * 20
#     df_future = df_future.append(df_add)

dfs_future = []
future_years = range(20,120, 20)
warming_decade = [0, 0.5, 1, 1.5, 2]
for i, y in zip(future_years, warming_decade):
    df_i = df.copy()
    df_i["T2"] = df_i["T2"] + y
    df_i["period"] = i
    dfs_future.extend([df_i])

##
future_runoff = pd.DataFrame(index=output_MATILDA[0].index)
for i, y in zip(dfs_future, future_years):
    output_MATILDA = MATILDA.MATILDA_submodules(i, parameter) # MATILDA model run + downscaling
    future_runoff[y] = output_MATILDA[0]["Q_Total"]

future_runoff.to_csv("/home/ana/Desktop/future_runoff.csv")

##
future_runoff = pd.read_csv("/home/ana/Desktop/future_runoff.csv")
future_runoff = future_runoff.set_index("TIMESTAMP")
future_runoff.index = pd.to_datetime(future_runoff.index)
future_runoff_monthly = future_runoff.resample("M").agg("sum")
future_runoff_monthly["month"] = future_runoff_monthly.index.month
future_runoff_monthly_mean = future_runoff_monthly.groupby(["month"]).mean()
future_runoff_monthly_mean = future_runoff_monthly_mean.reset_index()
future_runoff_monthly_mean.columns = ["month", "runoff_20", "runoff_40", "runoff_60", "runoff_80", "runoff_100"]

## Plot
barWidth = 0.15

ax = plt.subplot(111)
br1 = np.arange(len(future_runoff_monthly_mean["runoff_20"]))
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
br4 = [x + barWidth for x in br3]
br5 = [x + barWidth for x in br4]
# Make the plot
ax.bar(br1, future_runoff_monthly_mean["runoff_20"], color='#2b2d2f', width=barWidth, edgecolor='grey')
ax.bar(br2, future_runoff_monthly_mean["runoff_40"], color='#88CCEE', width=barWidth, edgecolor='grey')
ax.bar(br3, future_runoff_monthly_mean["runoff_60"], color='#DDCC77', width=barWidth, edgecolor='grey')
ax.bar(br4, future_runoff_monthly_mean["runoff_80"], color='#CC6677', width=barWidth, edgecolor='grey')
ax.bar(br5, future_runoff_monthly_mean["runoff_100"], color='#CC6677', width=barWidth, edgecolor='grey')
plt.ylabel('Runoff [mm]')
plt.xticks([r + barWidth for r in range(len(future_runoff_monthly_mean["month"]))], ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12"])
# Adding Xticks
plt.show()

##

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
