## import of necessary packages
from pathlib import Path
import sys
home = str(Path.home())
sys.path.append(home + '/Seafile/Ana-Lena_Phillip/data/scripts/Test_area/SPOTPY')
import mspot
from matilda_sample import df, obs, set_up_start, set_up_end, sim_start, sim_end,freq, area_cat, area_glac, ele_dat, ele_glac,ele_cat
import spotpy

## Creating an example file

# working_directory = home + "/Seafile/Ana-Lena_Phillip/data/"
# input_path_data = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/ERA5/Tien-Shan/At-Bashy/"
# input_path_observations = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/bash_kaindy/"
# data_csv = "no182_ERA5_Land_2000_202011_no182_41_75.9_fitted.csv"  # dataframe with columns T2 (Temp in Celsius), RRR (Prec in mm) and if possible PE (in mm)
# observation_data = "runoff_bashkaindy_04_2019-11_2020_temp_limit.csv"  # Daily Runoff Observations in mm
# output_path = working_directory + "input_output/output/" + data_csv[:15]
#
# df = pd.read_csv(input_path_data + data_csv)
# obs = pd.read_csv(input_path_observations + observation_data)
# obs["Qobs"] = obs["Qobs"] / 86400 * (46.232 * 1000000) / 1000  # Daten in mm, Umrechnung in m3/s
#
# set_up_start='2017-01-01 00:00:00'
# set_up_end='2018-12-31 23:00:00'
# sim_start='2017-01-01 00:00:00'
# sim_end='2018-11-01 23:00:00'
# freq="D"
# area_cat=7.53
# area_glac=2.95
# ele_dat=2550
# ele_glac=3957
# ele_cat=3830
#

##
setup = mspot.setup(set_up_start=set_up_start, set_up_end=set_up_end, sim_start=sim_start, sim_end=sim_end,
                    freq=freq, area_cat=area_cat, area_glac=area_glac, ele_dat=ele_dat, ele_glac=ele_glac,
                    ele_cat=ele_cat)

spot_setup = setup(df, obs)
sampler = spotpy.algorithms.mc(spot_setup, dbname='20210421_mpi_test', dbformat='None', parallel='mpi')
sampler.sample(10)

# best_summary = mspot.psample(df=df, obs=obs, rep=50, set_up_start='2018-01-01 00:00:00', set_up_end='2018-12-31 23:00:00',
#                        sim_start='2019-01-01 00:00:00', sim_end='2020-11-01 23:00:00', area_cat=46.232,
#                        area_glac=2.566, ele_dat=3864, ele_glac=4042, ele_cat=3360, parallel=True, ngs=2)
