##
from pathlib import Path; home = str(Path.home())
import xarray as xr
import pandas as pd
import matplotlib.pyplot as plt

working_directory = home + "/Seafile/Ana-Lena_Phillip/data/"
cosipy_nc = working_directory + "input_output/input/best_cosipyrun_no1/best_cosipyrun_no1_2011-18/best_cosipy_output_no1_2011-18.nc"
model_csv = working_directory + "scripts/Final_Model/Output/best_cosipy_output_no1_2011-18>>2020-09-16_14:32:14/model_output_2011-2018.csv"

cosipy = xr.open_dataset(cosipy_nc)
model = pd.read_csv(model_csv)
model.set_index("Unnamed: 0", inplace=True)
model.index = pd.to_datetime(model.index)

cosipy_runoff = cosipy.Q.mean(dim=["lat", "lon"])
cosipy_runoff_daily = cosipy_runoff.resample(time="D").mean(dim="time")
cosipy_smb = cosipy.surfMB.mean(dim=["lat", "lon"])
cosipy_melt = cosipy.surfM.mean(dim=["lat", "lon"])

print(str(sum(cosipy_runoff.values*1000))+" mm runoff from Cosipy")
print(str(sum(model["Q_Total"]))+ " mm total runoff from our model")
print(str(sum(cosipy_smb.values*1000))+" mm SMB from Cosipy")
print(str(sum(model["DDM_smb"]))+ " mm total SMB from our model")
print(str(sum(cosipy_melt.values*1000))+" mm melt from Cosipy")
print(str(sum(model["DDM_total_melt"]))+ " mm total melt from our model")

plt.plot(model.index.to_pydatetime(), model["Q_Total"])
plt.plot(model.index.to_pydatetime(), (cosipy_runoff_daily*1000))
plt.show()