"""
MATILDA (Modeling water resources in glacierized catchments)
"""
## Running all the required functions
import pandas as pd
import matplotlib.pyplot as plt
from MATILDA import dataformatting
from MATILDA import DDM
from MATILDA import HBV
from MATILDA import stats, plots

## Model configuration and data preprocessing
df = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/best_cosipyrun_no1/best_cosipyrun_no1_2011-18/best_cosipy_input_no1_2011-18.csv")
obs = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/glacierno1/hydro/daily_observations_2011-18.csv")

cal_period_start = '2011-01-01 00:00:00'
cal_period_end = '2011-12-31 23:00:00'
sim_period_start = '2012-01-01 00:00:00'
sim_period_end = '2018-12-31 23:00:00'

plot_frequency = "W"
plot_frequency_long = "Weekly"

df, obs = dataformatting.data_preproc(df, obs, cal_period_start, sim_period_end)
df_DDM = dataformatting.glacier_downscaling(df, height_diff=21, lapse_rate_temperature=-0.006, lapse_rate_precipitation=0)

## DDM model
degreedays_ds = DDM.calculate_PDD(df_DDM)
output_DDM = DDM.calculate_glaciermelt(degreedays_ds)

## HBV model
output_hbv = HBV.hbv_simulation(df, cal_period_start, cal_period_end)

## Output postprocessing
output = dataformatting.output_postproc(output_hbv, output_DDM, obs)

nash_sut = stats.NS(output["Qobs"], output["Q_Total"]) # Nash–Sutcliffe model efficiency coefficient
stats_output = stats.create_statistics(output)

plot_data = dataformatting.plot_data(output, plot_frequency, cal_period_start, sim_period_end)

## Plotting the output data
fig = plots.plot_meteo(plot_data, plot_frequency_long)
plt.show()

fig1 = plots.plot_runoff(plot_data, plot_frequency_long, nash_sut)
plt.show()

fig2 = plots.plot_hbv(plot_data, plot_frequency_long)
plt.show()
