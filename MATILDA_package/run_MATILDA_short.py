"""
MATILDA (Modeling water resources in glacierized catchments)
"""
## Running all the required functions
import pandas as pd
import matplotlib.pyplot as plt
from MATILDA import DDM # importing the DDM model functions
from MATILDA import HBV # importing the HBV model function
from MATILDA import stats, plots # importing functions for statistical analysis and plotting

## Model configuration
working_directory = "/home/ana/Seafile/Ana-Lena_Phillip/data/scripts/MATILDA_package/"
input_path_data = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/best_cosipyrun_no1/best_cosipyrun_no1_2011-18/"
input_path_observations = "/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/glacierno1/hydro/"

data_csv = "best_cosipy_input_no1_2011-18.csv"
observation_data = "daily_observations_2011-18.csv"

cal_period_start = '2011-01-01 00:00:00'
cal_period_end = '2011-12-31 23:00:00'
sim_period_start = '2012-01-01 00:00:00'
sim_period_end = '2018-12-31 23:00:00'

lapse_rate_temperature = -0.006
lapse_rate_precipitation = 0
height_diff = 21

plot_frequency = "W" # possible options are "D" (daily), "W" (weekly), "M" (monthly) or "Y" (yearly)
plot_frequency_long = "Weekly" # Daily, Weekly, Monthly or Yearly

## Data input preprocessing
df = pd.read_csv(input_path_data + data_csv)
obs = pd.read_csv(input_path_observations + observation_data)

df.set_index('TIMESTAMP', inplace=True)
df.index = pd.to_datetime(df.index)
df = df[cal_period_start: sim_period_end]
obs.set_index('Date', inplace=True)
obs.index = pd.to_datetime(obs.index)
obs = obs[cal_period_start: sim_period_end]

df_DDM = df.copy()
df_DDM["T2"] = df_DDM["T2"] + height_diff * float(lapse_rate_temperature)
df_DDM["RRR"] = df_DDM["RRR"] + height_diff * float(lapse_rate_precipitation)

## DDM model
degreedays_ds = DDM.calculate_PDD(df_DDM)
output_DDM = DDM.calculate_glaciermelt(degreedays_ds)

## HBV model
output_hbv = HBV.hbv_simulation(df, cal_period_start, cal_period_end)

## Output postprocessing
output = pd.concat([output_hbv, output_DDM], axis=1)
output = pd.concat([output, obs], axis=1)
output["Q_Total"] = output["Q_HBV"] + output["Q_DDM"]

nash_sut = stats.NS(output["Qobs"], output["Q_Total"]) # Nash–Sutcliffe model efficiency coefficient

## Statistical analysis
plot_data = output.resample(plot_frequency).agg(
    {"T2": "mean", "RRR": "sum", "PE": "sum", "Q_HBV": "sum", "Qobs": "sum", \
    "Q_DDM": "sum", "Q_Total": "sum", "HBV_AET": "sum", "HBV_snowpack": "mean", \
    "HBV_soil_moisture": "mean", "HBV_upper_gw": "mean", "HBV_lower_gw": "mean"})
plot_data = plot_data[cal_period_start: sim_period_end]

stats_output = stats.create_statistics(output)

## Plotting the output data
fig = plots.plot_meteo(plot_data, plot_frequency_long)
plt.show()

fig1 = plots.plot_runoff(plot_data, plot_frequency_long, nash_sut)
plt.show()

fig2 = plots.plot_hbv(plot_data, plot_frequency_long)
plt.show()

