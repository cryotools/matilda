import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from ConfigFile import plot_frequency, plot_save, output_path, area_name, compare_cosipy
# Statistical analysis of the output variables
def create_statistics(output_calibration):
    stats = output_calibration.describe()
    sum = pd.DataFrame(output_calibration.sum())
    sum.columns = ["sum"]
    sum = sum.transpose()
    stats = stats.append(sum)
    stats = stats.round(3)
    return stats

# Nash–Sutcliffe model efficiency coefficient
def NS(obs, model):
    return 1 - np.sum((obs-model)**2) / (np.sum((obs-obs.mean())**2))

# Plotting the meteorological parameters
def plot_meteo(plot_data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
    ax1.plot(plot_data.index.to_pydatetime(), (plot_data["T2"]), c="#d7191c")
    ax2.bar(plot_data.index.to_pydatetime(), plot_data["RRR"], width=10, color="#2c7bb6")
    ax3.plot(plot_data.index.to_pydatetime(), plot_data["PE"], c="#008837")
    plt.xlabel("Date", fontsize=9)
    ax1.grid(linewidth=0.25), ax2.grid(linewidth=0.25), ax3.grid(linewidth=0.25)
    ax1.set_title(plot_frequency +" mean temperature", fontsize=9)
    ax2.set_title(plot_frequency +" precipitation sum", fontsize=9)
    ax3.set_title(plot_frequency +" evapotranspiration sum", fontsize=9)
    ax1.set_ylabel("[°C]", fontsize=9)
    ax2.set_ylabel("[mm]", fontsize=9)
    ax3.set_ylabel("[mm]", fontsize=9)
    fig.suptitle(plot_frequency + " meteorological input parameters in " +str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4], size=14)
    plt.tight_layout()
    return fig

# Plotting the runoff
def plot_runoff(plot_data, nash_sut):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharey=True, figsize=(10,6))
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(plot_data.index.to_pydatetime(), plot_data['Qobs'], c="#0072B2", linewidth=1.2, label="Observations")
    ax1.plot(plot_data.index.to_pydatetime(), plot_data["Q_Total"], c="#D55E00", linewidth=1.2,  label="MATILDA")
    ax2 = plt.subplot(gs[1, :-1], sharey=ax1)
    ax2.plot(plot_data.index.to_pydatetime(), plot_data["Q_HBV"], c="#009E73", linewidth=1.2, label="HBV")
    ax3 = plt.subplot(gs[1:, -1], sharey=ax1)
    ax3.plot(plot_data.index.to_pydatetime(), plot_data["Q_DDM"], c="#CC79A7", linewidth=1.2, label="DDM")
    ax1.legend(), ax2.legend(), ax3.legend(),
    ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9)
    ax1.set_title(plot_frequency+ " runoff comparison of MATILDA and observations in "+ str(plot_data.index.values[1])[:4]+"-" \
              +str(plot_data.index.values[-1])[:4]+" for the "+area_name+" catchment", size=14)
    ax1.text(0.05, 0.95, 'NS coeff ' + str(round(nash_sut,2)),  transform=ax1.transAxes, fontsize=12,
        verticalalignment='top')
    plt.tight_layout()
    return fig

# Plotting the HBV output parameters
def plot_hbv(plot_data):
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, figsize=(10,6))
    ax1.plot(plot_data.index.to_pydatetime(), plot_data["HBV_AET"], "k")
    ax2.plot(plot_data.index.to_pydatetime(), plot_data["HBV_soil_moisture"], "k")
    ax3.plot(plot_data.index.to_pydatetime(), plot_data["HBV_snowpack"], "k")
    ax4.plot(plot_data.index.to_pydatetime(), plot_data["HBV_upper_gw"], "k")
    ax5.plot(plot_data.index.to_pydatetime(), plot_data["HBV_lower_gw"], "k")
    ax1.set_title("Actual evapotranspiration", fontsize=9)
    ax2.set_title("Soil moisture", fontsize=9)
    ax3.set_title("Water in snowpack", fontsize=9)
    ax4.set_title("Upper groundwater box", fontsize=9)
    ax5.set_title("Lower groundwater box", fontsize=9)
    plt.xlabel("Date", fontsize=9)
    ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9)
    ax4.set_ylabel("[mm]", fontsize=9), ax5.set_ylabel("[mm]", fontsize=9)
    fig.suptitle(plot_frequency +" output from the HBV model in the period "+ str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4], size=14)
    plt.tight_layout()
    return fig

def plot_cosipy(plot_data_cosipy, nash_sut, nash_sut_cosipy):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
    ax1.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["Qobs"], c="#0072B2", label="Observations")
    ax1.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["Q_Total"], c="#D55E00", alpha=0.7, label="MATILDA")
    ax1.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["Q_COSIPY"],  c="#CC79A7", alpha =0.7, label="COSIPY")
    ax2.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["DDM_total_melt"], c="#D55E00")
    ax2.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["COSIPY_melt"],  c="#CC79A7")
    ax3.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["DDM_smb"], c="#D55E00")
    ax3.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["COSIPY_smb"],  c="#CC79A7")
    ax1.set_title("Runoff comparison", fontsize=9)
    ax2.set_title("Total melt from DDM and COSIPY", fontsize=9)
    ax3.set_title("Surface mass balance from DDM and COSIPY", fontsize=9)
    plt.xlabel("Date", fontsize=9)
    ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9)
    ax1.legend(loc="upper right")
    fig.suptitle(plot_frequency +" output comparison from MATILDA and COSIPY in "+ str(plot_data_cosipy.index.values[1])[:4]+"-"+str(plot_data_cosipy.index.values[-1])[:4], size=14)
    ax1.text(0.05, 0.95, 'NS coeff ' + str(round(nash_sut, 2)) + "\nNS coeff COSIPY " \
             + str(round(nash_sut_cosipy, 2)), transform=ax1.transAxes, fontsize=8, verticalalignment='top')
    plt.tight_layout()
    return fig
