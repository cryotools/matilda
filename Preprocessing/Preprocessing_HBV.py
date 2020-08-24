from pathlib import Path; home = str(Path.home())      ### Zieht sich home vom system
#import sys; sys.path.append(home + '/Seafile/Ana-Lena_Phillip/scripts/')
import xarray as xr
import pandas as pd
import numpy as np
import datetime

# File organization
working_directory = '/Seafile/Ana-Lena_Phillip/data/'
output = home + working_directory + "HBV-Light/HBV-light_data/Glacier_No.1/Python/Data/"
output_hbv_py = home + "/Seafile/Ana-Lena_Phillip/data/scripts/LHMP/data/"

era5_file = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/20200810_Urumqi_ERA5_2000_2019_cosipy.csv"
runoff_observations = home + working_directory + "observations/glacierno1/hydro/dailyrunoff_2011-18_glacierno1.xls"

#Time slice
time_start = '2011-01-01 00:00:00'
time_end = '2018-12-31 23:00:00'

# Reading in the data and creating dataframes
# Units: Temp in K, Pev, ERA Runoff und TP in mm, Obs Runoff in m3/s
era5 = pd.read_csv(era5_file)
era5["T2"] = era5["T2"] - 273.15 # now °C
era5.set_index('TIMESTAMP', inplace=True)
era5.index = pd.to_datetime(era5.index)
era5 = era5["2011-01-01 00:00:00": "2018-12-31 23:00:00"]

### runoff is in m3/s
cols_runoff = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
skiprows_runoff = [0, 12, 13, 24, 25, 37, 38, 39, 40, 41, 42]
runoff = pd.read_excel(runoff_observations,
                       sheet_name=["2011","2012", "2013", "2014", "2015", "2016", "2017", "2018"],
                       usecols=cols_runoff,
                       skiprows=skiprows_runoff)

dfs = list()
for framename in runoff.keys():
    temp_df = runoff[framename]
    temp_df['Year'] = framename
    dfs.append(temp_df)
runoff = pd.concat(dfs)

runoff = runoff.rename(columns={'Day\Month': 'Day'}, index={'ONE': 'one'})
runoff["Year"] = runoff["Year"].astype(str).astype(int)

runoff = pd.melt(runoff, id_vars=["Day", "Year"], var_name=["Month"])

runoff["Month"] = [datetime.datetime.strptime(x, "%B").month for x in runoff["Month"]]
runoff["Date"] = pd.to_datetime(runoff[["Year", "Month", "Day"]], format="%Y-%m-%d",  errors='coerce')

runoff = runoff.drop(columns=["Day", "Month", "Year"])
runoff = runoff.dropna(subset=["Date"])
runoff = runoff.rename(columns={"value": "Q"})

# Runoff conversion to mm with the catchment size
runoff["Q"]= (runoff["Q"]*86400/3367000)*1000

##
# Daily values
era5_daily = era5.resample('D').agg({"T2":'mean',"RRR":'sum'})

# Estimation PE through a formula by Oudin et al. -> unit is mm / day
solar_constant = (1376 * 1000000) / 86400 # from 1376 J/m2s to MJm2d
extra_rad = 27.086217947590317
latent_heat_flux = 2.45
water_density = 1000

def calculate_pe(df):
        if df["T2"] + 5 > 0:
            return ((extra_rad/(water_density*latent_heat_flux))*((df["T2"]+5)/100)*1000)
        else:
            return 0

era5_daily["Pev_calculated"] = era5_daily.apply(calculate_pe, axis=1)

## Preparation for HBV Python Model
data = era5_daily[["T2", "RRR", "Pev_calculated"]]
data = data.rename(columns={"T2":"Temp", "RRR":"Prec", "Pev_calculated":"Evap"})
data.index.names = ['Date']

#data.to_csv(output_hbv_py + "data_urumqi.csv")

data_runoff = runoff.copy()
data_runoff = data_runoff.reset_index()
data_runoff.set_index('Date', inplace=True)
data_runoff = data_runoff.drop(columns=["index"])
data_runoff = data_runoff.rename(columns={"Q":"Qobs"})

data_runoff.to_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/observations/glacierno1/hydro/daily_observations_2011-18.csv")

runoff = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/observations/glacierno1/hydro/daily_observations_2011-18.csv")
runoff["Date"] = pd.to_datetime(runoff["Date"])
runoff = runoff.rename(columns={"Qobs":"Q"})


## Preparation for HBV Lite Model
ptq = era5_daily[["RRR", "T2"]]
ptq = ptq.rename(columns={"RRR":"P", "T2":"T"})
ptq["Date"] = ptq.index
ptq = pd.merge(ptq, runoff)

ptq["Date"] = ptq["Date"].apply(lambda x: x.strftime('%Y%m%d'))
ptq = ptq[["Date", "P", "T", "Q"]]
ptq["Q"][np.isnan(ptq["Q"])] = int(-9999)

ptq.to_csv(output + "ptq.txt", sep="\t", index=None)

evap_calc = era5_daily["Pev_calculated"]
#evap_calc = evap_calc.groupby([(evap_calc.index.month), (evap_calc.index.day)]).mean()
#evap_calc = evap_calc.drop((2, 29))
evap_calc.to_csv(output + "evap.txt", index=None, header=False)
