import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import glob
home = str(Path.home())
working_directory = home + '/Seafile/Tianshan_data/'

##
era5 = pd.read_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/book_chapter_data1981-2020_newTS.csv")
era5.set_index('time', inplace=True)
era5.index = pd.to_datetime(era5.index, utc=True)
#era5.index = era5.index.tz_localize('Asia/Bishkek')
era5 = era5.loc['2000-11-01 01:00:00':'2020-11-01 23:00:00']
#era5 = era5.sort_index()

total_precipitation = era5.tp.values #+ height_diff * lapse_rate_total_precipitation))         ### convert from m to mm
total_precipitation[total_precipitation < 0] = era5.tp.values[total_precipitation < 0]
total_precipitation[total_precipitation < 0] = 0
total_precipitation = total_precipitation * 1000
era5["tp"] = total_precipitation

#era5.to_csv("/home/ana/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/no182ERA5_Land_2000_2020.csv")

#era5 = era5.resample("D").agg({"t2m":"mean", "tp":"sum"})
time_start = '2019-01-01'  # longest timeseries of waterlevel sensor
time_end = '2019-12-31'
era_subset = era5.copy()
era_subset = era_subset[time_start:time_end]
#era_subset = era_subset.sort_index()

plt.plot(era_subset["t2m"])
plt.show()