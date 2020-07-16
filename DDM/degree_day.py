##
from pandas_degreedays import calculate_dd
import pandas as pd
import numpy as np
from pathlib import Path; home = str(Path.home())      ### Zieht sich home vom system
working_directory = home + '/Seafile/Ana-Lena_Phillip/data/scripts/pypdd'
input_path = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/'

input_file = '20200625_Umrumqi_ERA5_2011_2018_cosipy.csv'
input = input_path + input_file

#Time slice:
time_start = '2011-01-01T00:00'
time_end = '2018-12-31T23:00'

DS = pd.read_csv(input)
DS = DS.set_index('TIMESTAMP') # set Time as index
#DS.index = pd.to_datetime(DS.index)
DS = DS.assign(temp= DS.T2-273.15) # temp in degree celsius
## test pandas.degreedays
filename = '/home/ana/Downloads/temperature_sample.xls' # example from github to test
df_temp = pd.read_excel(filename)
df_temp = df_temp.set_index('datetime')
df_degreedays = calculate_dd(ts_temp2)

df_degreedays = calculate_dd(ts_temp)
# does not work and I don't know why. even the example df from github doesn't work

## selfmade degree days: input is a dataframe with time as index and temp in Celsius

def calculate_PDD(df):
    df.index = pd.to_datetime(df.index) # make sure it is datetime format
    # make sure temp unit is celsius
    temp_min = df.temp.resample('D').min()
    temp_max = df.temp.resample('D').max()
    temp_mean = df.temp.resample('D').mean()

    degreedays_df = {'temp_min': temp_min, 'temp_max': temp_max, "temp_avg": temp_mean}
    degreedays_df = pd.DataFrame(degreedays_df)
    degreedays_df["Date"] = degreedays_df.index

    # calculate the hydrological year
    def calc_hydrological_year(df):
        if 10 <= df.Date.month <= 12:
            water_year = df.Date.year + 1
            return water_year
        else:
            return df.Date.year

    degreedays_df['hydrological_year'] = degreedays_df.apply(lambda x: calc_hydrological_year(x), axis=1)
    degreedays_df.drop(["Date"], axis=1, inplace=True)

    # calculate the positive degree days
    def degree_days(df):
        if df["temp_avg"] > 0:
            return df["temp_avg"]
        else:
            return 0

    degreedays_df["PDD"] = degreedays_df.apply(degree_days, axis=1)
    degreedays_df["PDD_cum"] = degreedays_df["PDD"].cumsum()
    degreedays_df["PDD_cum_yearly"] = degreedays_df.groupby("hydrological_year")["PDD"].cumsum()

    return (degreedays_df)

degreedays_df =calculate_PDD(DS)

## glacier melt: M = KI*PDD + KS*PDD
# how to calculate KI and KS?

PARAMETERS = {
    'pdd_factor_snow':  3, # mm per day per Celsius
    'pdd_factor_ice':   8} # mm per day per Celsius

def calculate_glaciermelt(df):
    ice_melt = PARAMETERS['pdd_factor_ice'] * df.groupby("hydrological_year")["PDD"].sum()
    snow_melt = PARAMETERS['pdd_factor_snow'] * df.groupby("hydrological_year")["PDD"].sum()
    total_melt = snow_melt + ice_melt
    glacier_melt = pd.concat([ice_melt, snow_melt, total_melt], axis=1)
    glacier_melt.columns = ["ice_melt", "snow_melt", "total_melt"]
    return glacier_melt

glacier_melt = calculate_glaciermelt(degreedays_df) # output in mm
