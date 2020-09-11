# -*- coding: UTF-8 -*-
##
import sys
sys.path.extend(['/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Final_Model'])
#sys.path.extend(['/data/projects/ebaca/data/scripts/centralasiawaterresources/Final_Model'])
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ConfigFile import output_path, cal_exclude, sim_period_start, sim_period_end, cal_period_start, cal_period_end, plot_frequency, plot_save
from Scripts.Model import output

#output_yearly = output.resample("Y").agg({"T2":"mean", "RRR":"sum", "PE":"sum", "Q_HBV":"sum", "Qobs":"sum", "Q_DDM":"sum", \
#                                          "Q_Total":"sum"})
## Statistics
if cal_exclude == True:
    output_calibration = output[~(output.index < cal_period_end)]
else:
    output_calibration = output.copy()
stats = output_calibration.describe()
sum = pd.DataFrame(output_calibration.sum())
sum.columns = ["sum"]
sum = sum.transpose()
stats = stats.append(sum)
stats.to_csv(output_path + "model_stats_" +str(output_calibration.index.values[1])[:4]+"-"+str(output_calibration.index.values[-1])[:4]+".csv")
##
if plot_frequency == "daily":
    plot_data = output_calibration.copy()

elif plot_frequency == "monthly":
    plot_data = output_calibration.resample("M").agg({"T2":"mean", "RRR":"sum", "PE":"sum", "Q_HBV":"sum", "Qobs":"sum", "Q_DDM":"sum", \
                                           "Q_Total":"sum", "HBV_AET":"sum", "HBV_snowpack":"mean", "HBV_soil_moisture":"sum", \
                                          "HBV_upper_gw":"sum", "HBV_lower_gw":"sum"})

# Plot meteorological parameters
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
ax1.plot(plot_data.index.to_pydatetime(), (plot_data["T2"]-273.15), "red")
ax2.bar(plot_data.index.to_pydatetime(), plot_data["RRR"], width=10)
ax3.plot(plot_data.index.to_pydatetime(), plot_data["PE"], "green")
plt.xlabel("Date", fontsize=9)
ax1.grid(linewidth=0.25), ax2.grid(linewidth=0.25), ax3.grid(linewidth=0.25)
ax1.set_title("Monthly mean Temperature", fontsize=9)
ax2.set_title("Monthly Precipitation sum", fontsize=9)
ax3.set_title("Monthly Evapotranspiration sum", fontsize=9)
ax1.set_ylabel("[°C]", fontsize=9)
ax2.set_ylabel("[mm]", fontsize=9)
ax3.set_ylabel("[mm]", fontsize=9)
fig.suptitle("Meteorological input parameters in " +str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4], size=14)

if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "meteorological_data_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot runoff
fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,6))
gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0, :])
ax1.plot(plot_data.index.to_pydatetime(), plot_data['Qobs'], "k", label="Observations")
ax1.plot(plot_data.index.to_pydatetime(), plot_data["Q_Total"], "b", label="Total")
ax2 = plt.subplot(gs[1, :-1])
ax2.plot(plot_data.index.to_pydatetime(), plot_data["Q_HBV"], "g", alpha=0.8, label="HBV")
ax3 = plt.subplot(gs[1:, -1])
ax3.plot(plot_data.index.to_pydatetime(), plot_data["Q_DDM"], "r", alpha=0.8, label="DDM")
ax1.legend(), ax2.legend(), ax3.legend(),
ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9)
ax1.set_title("Daily runoff comparison of the model and observations in "+ str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4], size=14)

if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "model_runoff_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# Plot extra parameters, output of the HBV
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
fig.suptitle("Output from the HBV model in the period "+ str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4], size=14)
if plot_save == False:
	plt.show()
else:
	plt.savefig(output_path + "HBV_output_"+str(plot_data.index.values[1])[:4]+"-"+str(plot_data.index.values[-1])[:4]+".png")

# plt.legend(loc='upper center', bbox_to_anchor=(0.8, -0.21),
#           fancybox=True, shadow=True, ncol=5)
# if plot_save == False:
# 	plt.show()
# else:
# 	plt.savefig(output_path + "xtra_param+hbv_output"+str(sim_period_start[:4])+"-"+str(time_end[:4]+".png"))
print('Saved plots of meteorological and runoff data to disc')
print("End of model run")
print('---')
