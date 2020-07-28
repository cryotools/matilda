import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
import sys
home = str(Path.home()) # sys.path.append(home+'/source_code/')
import argparse
import time
import os
from misc_functions.plot_functions import plot_interesting_spatial, plot_cumulative, plot_all_timeline
from misc_functions.aggregation_functions import var_mean_space, var_sum_time_total

def plot_fields_spatial(input_file, start_date, end_date, name, integration):
    working_directory = input_file.split('/input/')[0]
    if name != None:
        plt_dir = working_directory + '/plots/' + name + '/'
    else:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt_dir = working_directory + '/plots/' + timestr + '/'
    plt_dir_all_spatial = plt_dir + 'all_variables_spatial/'
    plt_dir_all_time = plt_dir + 'all_variables_time/'
    os.mkdir(plt_dir)
    os.mkdir(plt_dir_all_spatial)
    os.mkdir(plt_dir_all_time)
    cosi = xr.open_dataset(input_file)
    if start_date != None:
        print('Slice dataset')
        cosi = cosi.sel(time=slice(start_date, end_date))

    cosi.HGT.values=cosi.HGT.values.astype(float)
    cosi.HGT.values[cosi.MASK.values!=1]=np.nan
   
    plt.figure(figsize=(16, 9))
    cosi.HGT.plot.pcolormesh('lon', 'lat')
    plt.savefig(plt_dir+'Elevation.png')
    plt.close()
    plot_all_timeline(cosi,plt_dir_all_time)
    plot_interesting_spatial(cosi, plt_dir_all_spatial, name)
    print('All plots done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot all fields of dataset spatially, line (mean space) plot and spatial (sum time) cumulative and altitude')
    parser.add_argument('-i', '-input_file', dest='input_file', help='Working directory (Input, Output and plots of Study area')
    parser.add_argument('-b', '-start_date', dest='start_date', help='Start date')
    parser.add_argument('-e', '-end_date', dest='end_date', help='End date')
    parser.add_argument('-n', '-name', dest='name', help='Name for plt dir and filename of plots')
    parser.add_argument('-t', '-integration', dest='integration', help='integration for time series')
    args = parser.parse_args()
    plot_fields_spatial(args.input_file, args.start_date, args.end_date, args.name, args.integration)
