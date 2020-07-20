##
import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path; home = str(Path.home())      ### Zieht sich home vom system
working_directory = home + '/Seafile/Ana-Lena_Phillip/data/scripts/pypdd'
input_path = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/'
static_path = home + '/Seafile/Ana-Lena_Phillip/data/input_output/static/'

input_file = '20200625_Umrumqi_ERA5_2011_2018_cosipy.nc'
static_file = 'Urumqi_static.nc'

input = input_path + input_file
static = static_path + static_file
output = working_directory + 'output/' + input_file.split('.')[0] + '_output.nc'

#Time slice:
time_start = '2011-01-01T00:00'
time_end = '2018-12-31T23:00'

##
DS = xr.open_dataset(input)
DS = DS.sel(time=slice(time_start, time_end))
temp = DS.T2.values - 273.15
temp_yearly = DS['T2'].resample(time="Y").mean(dim="time") - 273.15

DS = DS.assign(T2_C= DS.T2-273.15) # temp in degree celsius
stdv = DS['T2'].resample(time="Y").std(dim="time") # temp std for every year and every cell, needs to be in Kelvin

prec_yearly = DS['RRR'].resample(time="Y").sum(dim="time") # yearly sum for every cell
prec_yearly_mean = prec_yearly.mean() # mean of all the yearly sums
#prec = DS.RRR.values
##
mask = DS.MASK.values
temp_yearly = np.where(mask==1, temp_yearly, np.nan)
prec_yearly = np.where(mask==1, prec_yearly, np.nan)
#prec_yearly_mean = np.where(mask==1, prec_yearly_mean, np.nan) # does not work with mask
std = np.where(mask==1, std, np.nan)
##
from pypdd import PDDModel
pdd = PDDModel()
pdd_out = pdd(temp_yearly, prec_yearly, stdv)
pdd_out2 = pdd(temp_yearly, prec_yearly_mean, stdv)
##
hbv_light = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/HBV_Light/HBV-light_data/Glacier_No.1/Python/Glacier_Run/Results/Results.txt", sep="\t")
hbv_light_noglac = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/HBV_Light/HBV-light_data/Glacier_No.1/Python/Noglacier_Run/Results/Results.txt", sep="\t")
hbv = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/LHMP/output.csv")

pdd_runoff = np.where(np.isnan(pdd_out["runoff"]), 0, pdd_out["runoff"])
obs = hbv["Qobs"].sum()
sim = hbv["Qsim"].sum() + pdd_runoff.sum()*1000

# Yearly mean prec and yearly sum prec
diff = abs(pdd_out["runoff"] - pdd_out2["runoff"])

print(str(obs) + " Observation")
print(str(hbv["Qsim"].sum()) + " LHMP")
print(str(hbv_light_noglac["Qsim"].sum()) + " HBV Light No Glacier Run")
print(str(sum(hbv_light["Qsim"])) + " Glacier Run")
print(str(sim) + " LHMP + PYPDD")


#help(PDDModel)