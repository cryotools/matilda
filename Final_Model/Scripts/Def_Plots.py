# -*- coding: UTF-8 -*-
##
import sys
sys.path.extend(['/home/phillip/Seafile/Ana-Lena_Phillip/data/scripts/Final_Model'])
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from ConfigFile import output_path, time_start, time_end
from Scripts.Model import output

output_monthly = output.resample("M").agg({"T2":"mean", "RRR":"sum", "PE":"sum", "Q_HBV":"sum", "Qobs":"sum", "Q_DDM":"sum", \
                                           "Q_Total":"sum"})
output_yearly = output.resample("Y").agg({"T2":"mean", "RRR":"sum", "PE":"sum", "Q_HBV":"sum", "Qobs":"sum", "Q_DDM":"sum", \
                                          "Q_Total":"sum"})

##
# Plot meteorological parameters
fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True, figsize=(10,6))
ax1.plot(output_monthly.index.to_pydatetime(), (output_monthly["T2"]-273.15), "red")
ax2.bar(output_monthly.index.to_pydatetime(), output_monthly["RRR"], width=10)
ax3.plot(output_monthly.index.to_pydatetime(), output_monthly["PE"], "green")
plt.xlabel("Date", fontsize=9)
ax1.grid(linewidth=0.25), ax2.grid(linewidth=0.25), ax3.grid(linewidth=0.25)
ax1.set_title("Monthly mean Temperature", fontsize=9)
ax2.set_title("Monthly Precipitation sum", fontsize=9)
ax3.set_title("Monthly Evapotranspiration sum", fontsize=9)
ax1.set_ylabel("[°C]", fontsize=9)
ax2.set_ylabel("[mm]", fontsize=9)
ax3.set_ylabel("[mm]", fontsize=9)
fig.suptitle("Meteorological input parameters in " +str(time_start[:4])+"-"+str(time_end[:4]), size=14)
#plt.show()
plt.savefig(output_path + "meteorological_data_"+str(time_start[:4])+"-"+str(time_end[:4]+".png"))

# Plot Runoff
plt.figure(figsize=(10,6))
plt.plot(output.index.to_pydatetime(), output['Qobs'], "k", label="Observations")
plt.plot(output.index.to_pydatetime(), output["Q_Total"], "b", label="Total")
plt.plot(output.index.to_pydatetime(), output["Q_HBV"], "g--", alpha=0.5, label="HBV")
plt.plot(output.index.to_pydatetime(), output["Q_DDM"], "r--", alpha=0.5, label="DDM")
plt.legend()
plt.grid(linewidth=0.25)
plt.ylabel("[mm]", fontsize=9)
plt.xlabel("Date", fontsize=9)
plt.title("Comparison of the model runoff to the observation data in "+ str(time_start[:4])+"-"+str(time_end[:4]), size=14)
plt.show()
#plt.savefig(output_path + "model_runoff_"+str(time_start[:4])+"-"+str(time_end[:4]+".png"))

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10,6))
gs = gridspec.GridSpec(2, 2)
ax1 = plt.subplot(gs[0, :])
ax1.plot(output.index.to_pydatetime(), output['Qobs'], "k", label="Observations")
ax1.plot(output.index.to_pydatetime(), output["Q_Total"], "b", label="Total")
ax2 = plt.subplot(gs[1, :-1])
ax2.plot(output.index.to_pydatetime(), output["Q_HBV"], "g", alpha=0.8, label="HBV")
ax3 = plt.subplot(gs[1:, -1])
ax3.plot(output.index.to_pydatetime(), output["Q_DDM"], "r", alpha=0.8, label="DDM")
ax1.legend(), ax2.legend(), ax3.legend(),
ax1.set_ylabel("[mm]", fontsize=9), ax2.set_ylabel("[mm]", fontsize=9), ax3.set_ylabel("[mm]", fontsize=9)
ax1.set_title("Daily runoff comparison of the model and observations in "+ str(time_start[:4])+"-"+str(time_end[:4]), size=14)
#plt.show()
plt.savefig(output_path + "model_runoff_"+str(time_start[:4])+"-"+str(time_end[:4]+".png"))
print('Saved plots of meteorological and runoff data to disc')
print("End of model run")
print('---')
