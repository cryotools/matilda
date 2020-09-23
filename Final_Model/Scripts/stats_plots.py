import pandas as pd
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
    return stats

# Plotting the meteorological parameters
def plot_meteo(plot_data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
    ax1.plot(plot_data.index.to_pydatetime(), (plot_data["T2"]-273.15), "red")
    ax2.bar(plot_data.index.to_pydatetime(), plot_data["RRR"], width=10)
    ax3.plot(plot_data.index.to_pydatetime(), plot_data["PE"], "green")
    plt.xlabel("Date", fontsize=9)
    ax1.grid(linewidth=0.25), ax2.grid(linewidth=0.25), ax3.grid(linewidth=0.25)
    ax1.set_title(plot_frequency +" mean temperature", fontsize=9)
    ax2.set_title(plot_frequency +" precipitation sum", fontsize=9)
    ax3.set_title(plot_frequency +" evapotranspiration sum", fontsize=9)
    ax1.set_ylabel("[°C]", fontsize=9)
    ax2.set_ylabel("[mm]", fontsize=9)
    ax3.set_ylabel("[mm]", fontsize=9)
    fig.suptitle("Meteorological input parameters in " +str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4], size=14)
    return fig

# Plotting the runoff
def plot_runoff(plot_data):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharey=True, figsize=(10,6))
    gs = gridspec.GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, :])
    ax1.plot(plot_data.index.to_pydatetime(), plot_data['Qobs'], "k", linewidth=1.2, label="Observations")
    ax1.plot(plot_data.index.to_pydatetime(), plot_data["Q_Total"], "b", linewidth=1.2, alpha=0.6, label="Model Total")
    ax2 = plt.subplot(gs[1, :-1], sharey=ax1)
    ax2.plot(plot_data.index.to_pydatetime(), plot_data["Q_HBV"], "g", linewidth=1.2, label="HBV")
    ax3 = plt.subplot(gs[1:, -1], sharey=ax1)
    ax3.plot(plot_data.index.to_pydatetime(), plot_data["Q_DDM"], "r", linewidth=1, label="DDM")
    ax1.legend(), ax2.legend(), ax3.legend(),
    ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9)
    ax1.set_title(plot_frequency+ " runoff comparison of the model and observations in "+ str(plot_data.index.values[1])[:4]+"-" \
              +str(plot_data.index.values[-1])[:4]+" for the "+area_name+" catchment", size=14)
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
    return fig

def plot_cosipy(plot_data_cosipy):
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
    ax1.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["Qobs"], "k", label="Observations")
    ax1.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["Q_Total"], "b", label="Model")
    ax1.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["Q_COSIPY"], "r", label="COSIPY")
    ax2.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["DDM_total_melt"], "b")
    ax2.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["COSIPY_melt"], "r")
    ax3.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["DDM_smb"], "b")
    ax3.plot(plot_data_cosipy.index.to_pydatetime(), plot_data_cosipy["COSIPY_smb"], "r")
    ax1.set_title("Runoff comparison", fontsize=9)
    ax2.set_title("Total melt from DDM and COSIPY", fontsize=9)
    ax3.set_title("Surface mass balance from DDM and COSIPY", fontsize=9)
    plt.xlabel("Date", fontsize=9)
    ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9)
    ax1.legend()
    fig.suptitle(plot_frequency +" output comparison from the model and COSIPY in "+ str(plot_data_cosipy.index.values[1])[:4]+"-"+str(plot_data_cosipy.index.values[-1])[:4], size=14)
    return fig
