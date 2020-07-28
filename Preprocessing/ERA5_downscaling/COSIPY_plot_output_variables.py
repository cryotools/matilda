import numpy as np
import xarray as xr
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import sys
home = str(Path.home())
import argparse
import time
import os
from misc_functions.plot_functions import plot_interesting_spatial, plot_cumulative, plot_all_timeline, plot_bar_water_year, plot_bar_monthly
from misc_functions.aggregation_functions import var_mean_space, var_sum_time_total

def plot_fields_spatial(input_file, start_date, end_date, name, integration, all):
    working_directory = input_file.split('/output/')[0]
    if integration == None:
        integration = 'h'
    if name != None:
        plt_dir = working_directory + '/plots/' + name + '/'
    else:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        plt_dir = working_directory + '/plots/' + timestr + '/'
#    else:
#        plt_dir = working_directory + '/plots/comparison_cosi_stakes/' + os.path.splitext((input_file.split('/output/')[1]))[0]
#
    print(plt_dir)
    print(integration)
    plt_dir_all_spatial = plt_dir + 'all_variables_spatial/'
    plt_dir_all_time = plt_dir + 'all_variables_time/'

    os.mkdir(plt_dir)
    cosi = xr.open_dataset(input_file)

    if start_date != None:
        print('Slice dataset')
        cosi = cosi.sel(time=slice(start_date, end_date))

    list_of_variables_to_drop = ['HGT', 'T2', 'U2', 'RH2', 'U2', 'PRES', 'G', 'RRR', 'N', 'RAIN', 'SNOWFALL', 'NLAYERS', 'ME', 'EVAPORATION', 'DEPOSITION', 'CONDENSATION', 'REFREEZE', 'surfM', 'Z0']
    #list_of_variables_to_drop= ['HGT', 'T2', 'RH2', 'U2', 'PRES', 'G', 'N', 'RRR', 'NLAYERS']
    #cosi = cosi.drop(labels=list_of_variables_to_drop)

    cosi.HGT.values=cosi.HGT.values.astype(float)
    cosi.HGT.values[cosi.MASK.values!=1]=np.nan
    plt.figure(figsize=(16, 9))
    cosi.HGT.plot.pcolormesh('lon', 'lat')
    plt.savefig(plt_dir+'Elevation.png')

    mb_mean = var_mean_space(cosi.MB)
    surfmb_mean = var_mean_space(cosi.surfMB)
    intmb_mean = var_mean_space(cosi.intMB)
    plot_cumulative(mb_mean,plt_dir)
    plot_cumulative(surfmb_mean,plt_dir)
    plot_cumulative(intmb_mean,plt_dir)
    mb_sum_time =  var_sum_time_total(cosi.MB)
    plot_bar_water_year(mb_mean, plt_dir)
    plot_bar_water_year(surfmb_mean, plt_dir)
    plot_bar_monthly(mb_mean, plt_dir)
    
    min = np.nanmin(mb_sum_time)
    max = np.nanmax(mb_sum_time)
    
    mb_sum_time.values[cosi.MASK.values!=1]=np.nan
    cosi.HGT.values=cosi.HGT.values.astype(float)
    cosi.HGT.values[cosi.MASK.values!=1]=np.nan
   
    plt.figure(figsize=(16, 9))
    mb_sum_time.plot.pcolormesh('lon', 'lat')
    plt.savefig(plt_dir+'cumulative_total_sum_mesh.png')
    plt.close()

    plt.figure(figsize=(16, 9))
    cosi.HGT.plot.pcolormesh('lon', 'lat')
    plt.savefig(plt_dir+'Elevation.png')
    plt.close()

    if all != None:
      os.mkdir(plt_dir_all_spatial)
      os.mkdir(plt_dir_all_time)
      plot_all_timeline(cosi,plt_dir_all_time,integration)
      plot_interesting_spatial(cosi, plt_dir_all_spatial, name)

    print("All plots done")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot all fields of dataset spatially, line (mean space) plot and spatial (sum time) cumulative and altitude')
    parser.add_argument('-i', '-input_file', dest='input_file', help='Working directory (Input, Output and plots of Study area')
    parser.add_argument('-b', '-start_date', dest='start_date', help='Start date')
    parser.add_argument('-e', '-end_date', dest='end_date', help='End date')
    parser.add_argument('-n', '-name', dest='name', help='Name for plt dir and filename of plots')
    parser.add_argument('-t', '-integration', dest='integration', help='integration for time series')
    parser.add_argument('-a', '-all', dest='all', help='do all spatial and temporal plots')
    args = parser.parse_args()
    plot_fields_spatial(args.input_file, args.start_date, args.end_date, args.name, args.integration, args.all)
