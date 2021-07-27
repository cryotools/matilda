## Running all the required functions
from pathlib import Path; home = str(Path.home())
from datetime import datetime
import os
#import warnings
#warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MATILDA_slim import MATILDA

## Model configuration
# Directories
cmip_data = home + "/Seafile/Tianshan_data/CMIP/CMIP6/all_models/Kysylsuu/"
glacier_profile = pd.read_csv(home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/karabatkak_glacier_profile_tricht.csv")
output_path = home + "/Seafile/Ana-Lena_Phillip/data/input_output/output/new_deltaH/Kashkator"

cmip_mean = pd.read_csv(cmip_data + "CMIP6_mean_42.25-78.25_1980-01-01-2100-12-31_downscaled.csv")
scenarios = ["cmip_2_6", "cmip_4_5", "cmip_8_5"]
cmip_2_6 = cmip_mean[["time", "temp_26", "prec_26"]]
cmip_4_5 = cmip_mean[["time","temp_45", "prec_45"]]
cmip_8_5 = cmip_mean[["time","temp_85", "prec_85"]]

cmip_dfs = [cmip_2_6, cmip_4_5, cmip_8_5]

for i in cmip_dfs:
    i.columns = ['TIMESTAMP', 'T2', 'RRR']

##
scen = "cmip_4_5"

for df, scen in zip(cmip_dfs, scenarios):
    parameter = MATILDA.MATILDA_parameter(cmip_dfs[1], set_up_start='2015-01-01 12:00:00',
                                          set_up_end='2020-12-31 12:00:00',
                                          sim_start='2021-01-01 12:00:00', sim_end='2100-12-31 12:00:00', freq="Y",
                                          lat=42.25,
                                          area_cat=7.527, area_glac=2.271, ele_dat=2550, ele_glac=3830, ele_cat= 3830, lr_temp=-0.005936,
                                          lr_prec=-0.0002503, TT_snow=0.354, TT_rain=0.5815, CFMAX_snow=4.824,
                                          CFMAX_ice=5.574,CFR_snow=0.08765, CFR_ice=0.01132, BETA=2.03, CET=0.0471,
                                          FC=462.5, K0=0.03467, K1=0.0544, K2=0.1277, LP=0.4917, MAXBAS=2.494, PERC=1.723,
                                          UZL=413.0, PCORR=1.19, SFCF=0.874, CWH=0.011765)
    df_preproc = MATILDA.MATILDA_preproc(cmip_dfs[1], parameter)
    output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, glacier_profile=glacier_profile)
    output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
    MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path)
    output_path2 = output_path + "_" + str(scen)
    MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path2) # save regular MATILDA run

    # Plot
    if "Q_DDM" in output_MATILDA[0].columns:
        plot_data = output_MATILDA[0].resample(parameter.freq).agg(
            {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", \
             "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
             "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"}, skipna=False)

    plot_area = output_MATILDA[4].copy()
    plot_area['time'].iloc[0] = float(2020)
    plot_area['time'] = pd.to_numeric(plot_area['time'], errors='coerce')
    plot_area["time"] = pd.to_datetime(plot_area["time"], format='%Y')

    plot_data["plot"] = 0
    plot_data.loc[plot_data.isnull().any(axis=1), :] = np.nan
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 4.5), gridspec_kw={'width_ratios': [2.75, 1]})
    ax1.fill_between(plot_data.index.to_pydatetime(), plot_data["plot"], plot_data["Q_HBV"], color='#56B4E9',
                     alpha=.75, label="")
    if "Q_DDM" in plot_data.columns:
        ax1.plot(plot_data.index.to_pydatetime(), plot_data["Q_Total"], c="k", label="", linewidth=0.75, alpha=0.75)
        ax1.fill_between(plot_data.index.to_pydatetime(), plot_data["Q_HBV"], plot_data["Q_Total"], color='#CC79A7',
                         alpha=.75, label="")
    ax1.set_ylabel("Runoff [mm]", fontsize=9)
    ax2.plot(plot_area["time"], plot_area["glacier_area"], color="black")
    ax2.set_ylabel("Glacier area [km2]", fontsize=9)
    if str(plot_data.index.values[1])[:4] == str(plot_data.index.values[-1])[:4]:
        plt.suptitle(
            parameter.freq_long + " MATILDA simulation for the period " + str(plot_data.index.values[-1])[:4],
            size=14)
    else:
        plt.suptitle(parameter.freq_long + " MATILDA simulation for the period " + str(plot_data.index.values[0])[
                                                                                   :4] + "-" + str(
            plot_data.index.values[-1])[:4], size=14)
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.02),
               bbox_transform=plt.gcf().transFigure)
    plt.tight_layout()
    plt.savefig("/home/ana/Desktop/" + str(scen) + "runoff_glac_area.png")

    #
    plot_annual_data = output_MATILDA[0].copy()
    plot_annual_data["month"] = plot_annual_data.index.month
    plot_annual_data["day"] = plot_annual_data.index.day
    plot_annual_data = plot_annual_data.groupby(["month", "day"]).mean()
    plot_annual_data["date"] = pd.date_range(parameter.sim_start, freq='D', periods=len(plot_annual_data)).strftime(
        '%Y-%m-%d')
    plot_annual_data = plot_annual_data.set_index(plot_annual_data["date"])
    plot_annual_data.index = pd.to_datetime(plot_annual_data.index)

    fig, (ax1, ax2) = plt.subplots(2, sharex=True, figsize=(10, 6))
    ax1.plot(plot_annual_data.index.to_pydatetime(), (plot_annual_data["T2"]), c="#d7191c")
    ax2.bar(plot_annual_data.index.to_pydatetime(), plot_annual_data["RRR"], width=10, color="#2c7bb6")
    plt.xlabel("Date", fontsize=9)
    ax1.grid(linewidth=0.25), ax2.grid(linewidth=0.25)
    ax1.set_title("Mean temperature", fontsize=9)
    ax2.set_title("Precipitation sum", fontsize=9)
    ax1.set_ylabel("[°C]", fontsize=9)
    ax2.set_ylabel("[mm]", fontsize=9)
    fig.suptitle("Annual cycle for the years" + str(plot_data.index.values[0])[:4] + " to " + str(plot_data.index.values[-1])[:4], size=14)
    plt.tight_layout()
    fig.set_size_inches(10, 6)
    plt.savefig("/home/ana/Desktop/" + str(scen) + "annual_cycle_meterological_data.png")

