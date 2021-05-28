from pathlib import Path; import sys; import numpy as np; import xarray as xr; import pandas as pd
home = str(Path.home()); sys.path.append('/home/anz/Seafile/work/software/python_source_code/pre_and_postprocessing_COSIPY/Anselm/')
from misc_functions.inspect_values import calculate_r2_and_rmse
from misc_functions.calculate_parameters import calculate_water_year, calculate_season
from misc_functions.aggregation_functions import var_sum_time, var_mean_time

def correlate_ami(cosi_yearly, ind):
    start_date = max(cosi_yearly.time[0], ind.time[0])
    end_date = min(cosi_yearly.time[-1], ind.time[-1])
    cosi_yearly = cosi_yearly.sel(time=slice(start_date, end_date))
    ind = ind.sel(time=slice(start_date, end_date))
    dez_r2, dez_rmse, dez_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Dez'], 'cosi AMI Dez')
    jan_r2, jan_rmse, jan_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Jan'], 'cosi AMI Jan')
    feb_r2, feb_rmse, feb_s_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Feb'], 'cosi AMI Feb')
    djf_r2, djf_rmse, djf_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['DJF'], 'cosi AMI DFJ')

def correlate_summer_indexes(cosi_yearly, ind):
    start_date = max(cosi_yearly.time[0], ind.time[0])
    end_date = min(cosi_yearly.time[-1], ind.time[-1])
    cosi_yearly = cosi_yearly.sel(time=slice(start_date, end_date))
    ind = ind.sel(time=slice(start_date, end_date))
    jun_r2, jun_rmse, jun_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Jun'], 'cosi ' + str(ind.long_name) + 'Jun')
    jul_r2, jul_rmse, jul_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Jul'], 'cosi ' + str(ind.long_name) + 'Jul')
    aug_r2, aug_rmse, aug_s_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Aug'], 'cosi ' + str(ind.long_name) + 'Aug')
    sep_r2, dep_rmse, sep_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Sep'], 'cosi ' + str(ind.long_name) + 'Sep')
    jja_r2, jja_rmse, jja_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['JJA'], 'cosi ' + str(ind.long_name) + 'JJA')
    jjas_r2, jjas_rmse, jjas_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['JJAS'], 'cosi ' + str(ind.long_name) + 'JJAS')
    return np.round(jun_r2.values,2), np.round(jul_r2.values,2), np.round(aug_r2.values,2), np.round(sep_r2.values,2), np.round(jja_r2.values,2), np.round(jjas_r2.values,2)

def correlate_monthly_indexes(cosi, ind, season_start, season_end, new_year=False):
    #breakpoint()
    start_date = max(cosi.time[0], ind.time[0])
    end_date = min(cosi.time[-1], ind.time[-1])
    cosi = cosi.sel(time=slice(start_date, end_date))
    ind = ind.sel(time=slice(start_date, end_date))
    cosi_yearly = var_sum_time(cosi, 'y')
    ind_yearly = var_mean_time(ind, 'y')
    ind = ind.sel(time=slice(start_date, end_date))
    seasons_years, seasons_values = calculate_season(ind, 'mean', season_start, season_end, new_year=False)
    seasons = xr.Dataset()
    #seasons.coords['time'] = cosi_yearly.time.values
    seasons.coords['time'] = pd.date_range(start=str(seasons_years[0]), periods=len(seasons_values), freq = 'Y')
    seasons['index_season'] = (('time'), seasons_values)
    yearly_r2, y_rmse, y_p = calculate_r2_and_rmse(cosi_yearly.MB, ind_yearly, 'cosi ' + str(ind.long_name) + ' yearly')
    monthly_r2, m_rmse, m_p = calculate_r2_and_rmse(cosi.MB, ind, 'cosi ' + str(ind.long_name) + ' monthly')
    season_r2, s_rmse, s_p = calculate_r2_and_rmse(cosi_yearly.MB, seasons['index_season'], 'cosi_yearly_ ' + str(ind.long_name) + ' season')

    # jun_r2, jun_rmse, jun_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Jun'], 'cosi ' + str(ind.long_name) + 'Jun')
    # jul_r2, jul_rmse, jul_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Jul'], 'cosi ' + str(ind.long_name) + 'Jul')
    # aug_r2, aug_rmse, aug_s_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Aug'], 'cosi ' + str(ind.long_name) + 'Aug')
    # sep_r2, dep_rmse, sep_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['Sep'], 'cosi ' + str(ind.long_name) + 'Sep')
    # jja_r2, jja_rmse, jja_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['JJA'], 'cosi ' + str(ind.long_name) + 'JJA')
    # jjas_r2, jjas_rmse, jjas_p = calculate_r2_and_rmse(cosi_yearly.MB, ind['JJAS'], 'cosi ' + str(ind.long_name) + 'JJAS')
    # return np.round(jun_r2,2), np.round(jul_r2,2), np.round(aug_r2,2), np.round(sep_r2,2), np.round(jja_r2,2), np.round(jjas_r2,2)