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
## MATILDA initial run
df = pd.read_csv(home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182_ERA5_Land_1982_2020_41_75.9_fitted2AWS.csv")
obs_poly = pd.read_csv(home + "/Seafile/Tianshan_data/Gauging_station_Bash-Kaingdy/preprocessed/discharge_bahskaingdy_polyfitted_2019-04_11-2020.csv")
obs_double = pd.read_csv(home + "/Seafile/Tianshan_data/Gauging_station_Bash-Kaingdy/preprocessed/discharge_bahskaingdy_double-polyfitted_2019-04_11-2020.csv")
glacier_profile = pd.read_csv(home + "/Seafile/Masterarbeit/Data/glacier_profile.txt")

df.columns = ["TIMESTAMP", "T2", "RRR"]
obs_poly.columns = ['Date', 'Qobs']
obs_double.columns = ['Date', 'Qobs']

parameter = MATILDA.MATILDA_parameter(df, set_up_start='2018-01-01 12:00:00', set_up_end='2018-12-31 12:00:00',
                                      sim_start='2019-01-01 12:00:00', sim_end='2020-12-31 12:00:00', freq="D",
                                      lat=41, area_cat=46.23, area_glac=2.566, ele_dat=2250, ele_glac=4035, ele_cat=3485,
                                      CFMAX_ice=5, CFMAX_snow=2.5, BETA=1, CET=0.15, FC=200, K0=0.055, K1= 0.055, K2=0.04,
                                      LP=0.7, MAXBAS=2, PERC=2.5, UZL=60, TT_snow=-0.5, TT_rain=2, SFCF=0.7, CFR_ice=0.05,
                                      CFR_snow= 0.05, CWH=0.1)
df_preproc, obs_preproc = MATILDA.MATILDA_preproc(df, parameter, obs=obs_poly)
output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, obs=obs_preproc, glacier_profile=glacier_profile)
output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)

output_MATILDA[6].show()

output_path = "/home/ana/Desktop/"
MATILDA_save_output(output_MATILDA, parameter, output_path)



## Model configuration
# Directories
cmip_data = home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/input/bashkaingdy/met/cmip6/"
output_path = home + "/Seafile/Ana-Lena_Phillip/data/input_output/output/new_deltaH/Bash_Kaindy"
glacier_profile = pd.read_csv(home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/bash-kaindy_glacier_profile_glabtop.csv")


cmip_mean = pd.read_csv(cmip_data + "CMIP6_mean_41-75.9_1980-01-01-2100-12-31_downscaled.csv")
scenarios = ["cmip_2_6", "cmip_4_5", "cmip_8_5"]
cmip_2_6 = cmip_mean[["time", "temp_26", "prec_26"]]
cmip_4_5 = cmip_mean[["time","temp_45", "prec_45"]]
cmip_8_5 = cmip_mean[["time","temp_85", "prec_85"]]

cmip_dfs = [cmip_2_6, cmip_4_5,  cmip_8_5]

for i in cmip_dfs:
    i.columns = ['TIMESTAMP', 'T2', 'RRR']

##
parameter = MATILDA.MATILDA_parameter(cmip_4_5, set_up_start='2015-01-01 12:00:00', set_up_end='2020-12-31 12:00:00',
                                      sim_start='2021-01-01 12:00:00', sim_end='2100-12-31 12:00:00', freq="M",
                                      lat=41, area_cat=46.23, area_glac=2.566, ele_dat=2550, ele_glac=4035, ele_cat=3485,
                                      CFMAX_ice=5, CFMAX_snow=2.5, BETA=1, CET=0.15, FC=200, K0=0.055, K1= 0.055, K2=0.04,
                                      LP=0.7, MAXBAS=2, PERC=2.5, UZL=60, TT_snow=-0.5, TT_rain=2, SFCF=0.7, CFR_ice=0.05,
                                      CFR_snow= 0.05, CWH=0.1)
df_preproc = MATILDA.MATILDA_preproc(cmip_4_5, parameter)
output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, glacier_profile=glacier_profile)
output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
MATILDA.MATILDA_save_output(output_MATILDA, parameter, "/home/ana/Desktop/") # save regular MATILDA run


##
for df, scen in zip(cmip_dfs, scenarios):
    parameter = MATILDA.MATILDA_parameter(df, set_up_start='2015-01-01 12:00:00', set_up_end='2020-12-31 12:00:00',
                                          sim_start='2021-01-01 12:00:00', sim_end='2100-12-31 12:00:00', freq="Y",
                                          lat=41, area_cat=46.23, area_glac=2.566, ele_dat=2250, ele_glac=4035, ele_cat=3485,
                                          CFMAX_ice=5, CFMAX_snow=2.5, BETA=1, CET=0.15, FC=200, K0=0.055, K1= 0.055, K2=0.04,
                                          LP=0.7, MAXBAS=2, PERC=2.5, UZL=60, TT_snow=-0.5, TT_rain=2, SFCF=0.7, CFR_ice=0.05,
                                          CFR_snow= 0.05, CWH=0.1)
    df_preproc = MATILDA.MATILDA_preproc(df, parameter)
    output_MATILDA = MATILDA.MATILDA_submodules(df_preproc, parameter, glacier_profile=glacier_profile)
    output_MATILDA = MATILDA.MATILDA_plots(output_MATILDA, parameter)
    MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path=output_path)
    output_path2 = output_path + "_" + str(scen)
    MATILDA.MATILDA_save_output(output_MATILDA, parameter, output_path2) # save regular MATILDA run

    # Plot
    if "Q_DDM" in output_MATILDA[0].columns:
        plot_data = output_MATILDA[0].resample(parameter.freq).agg(
            {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", \
             "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
             "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"}, skipna=False)

    plot_area = output_MATILDA[4].copy()
    plot_area['time'].iloc[0] == float(2020)
    plot_area['time'] = plot_area['time']
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


## Moving average test
output = pd.read_csv(home + "/Seafile/Ana-Lena_Phillip/data/input_output/output/bash_kaindy_cmip_2_62021_2100_2021-06-15_19:48:58/model_output_2021-2100.csv").set_index("TIMESTAMP")
output.index = pd.to_datetime(output.index)

plot_data = output.resample("Y").agg({"Q_Total": "sum"}, skipna=False)
plot_data['Q10'] = plot_data.Q_Total.rolling(10, min_periods=1).mean()

plt.plot(plot_data.index.to_pydatetime(), (plot_data["Q_Total"]), c="black")
plt.plot(plot_data.index.to_pydatetime(), (plot_data["Q10"]), c="#d7191c")
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.legend(labels =['MATILDA Output', '10-years Q'], fontsize=14)
plt.title('Yearly runoff in Bash Kaindy for the CMIP 2.6 scenario', fontsize=14)
plt.xlabel('Year', fontsize=10)
plt.ylabel('Discharge [mm]', fontsize=10)
plt.show()