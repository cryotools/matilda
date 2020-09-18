##
from pathlib import Path; home = str(Path.home())
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


working_directory = home + "/Seafile/Ana-Lena_Phillip/data/"
cosipy_nc = working_directory + "input_output/input/best_cosipyrun_no1/best_cosipyrun_no1_2011-18/best_cosipy_output_no1_2011-18.nc"
model_csv = working_directory + "scripts/Final_Model/Output/best_cosipy_output_no1_2011-18>>2020-09-16_14:32:14/model_output_2011-2018.csv"

cosipy = xr.open_dataset(cosipy_nc)
model = pd.read_csv(model_csv)
model.set_index("Unnamed: 0", inplace=True)
model.index = pd.to_datetime(model.index)

cosipy_runoff = cosipy.Q.mean(dim=["lat", "lon"])
cosipy_runoff_daily = cosipy_runoff.resample(time="D").sum(dim="time")
cosipy_smb = cosipy.surfMB.mean(dim=["lat", "lon"])
cosipy_melt = cosipy.surfM.mean(dim=["lat", "lon"])

print(str(sum(cosipy_runoff.values*1000))+" mm runoff from Cosipy")
print(str(sum(model["Q_Total"]))+ " mm total runoff from our model")
print(str(sum(cosipy_smb.values*1000))+" mm SMB from Cosipy")
print(str(sum(model["DDM_smb"]))+ " mm total SMB from our model")
print(str(sum(cosipy_melt.values*1000))+" mm melt from Cosipy")
print(str(sum(model["DDM_total_melt"]))+ " mm total melt from our model")

fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(10, 6))
ax1.plot(model.index.to_pydatetime(), model["Q_Total"], "k")
ax1.plot(model.index.to_pydatetime(), (cosipy_runoff_daily*1000, "b")
ax2.plot(model.index.to_pydatetime(), model["DDM_smb"], "k")
ax2.plot(model.index.to_pydatetime(), (cosipy_smb*1000), "b")
ax3.plot(model.index.to_pydatetime(), model["DDM_total_melt"], "k")
ax3.plot(model.index.to_pydatetime(), (cosipy_melt*1000), "b")
ax1.set_title("Runoff", fontsize=9)
ax2.set_title("Surface mass balance", fontsize=9)
ax3.set_title("Melt", fontsize=9)
plt.xlabel("Date", fontsize=9)
plt.show()
