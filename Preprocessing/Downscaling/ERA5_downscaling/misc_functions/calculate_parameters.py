import numpy as np
from fundamental_physical_constants import teten_a1, teten_a3, teten_a4, zero_temperature

def calculate_ew_sonntag(T):
  if T>=273.16:
      # over water
    Ew = 6.112 * np.exp((17.67*(T-273.16)) / ((T-29.66)))
  else:
    Ew = 6.112 * np.exp((22.46*(T-273.16)) / ((T-0.55)))
  return Ew

def calculate_ew(T):
    ### a4 K, a1 Pa, a3 ( )
    ew = teten_a1 * np.exp(teten_a3 * (T - zero_temperature) / (T - teten_a4))
    return ew   ### Pa

def calculate_qs(Ew, P):
    qs = (611.21 / 1000) * (Ew / (0.01 * P - (1 - (611.21 / 1000)) * Ew))
    return qs

def calculte_mixing_radio(q2):
    mixing_ratio = q2/(1-q2)
    return mixing_ratio

def calculate_mixing_ratio_rh2(rh2, Ew, p):
    mixing_ratio = rh2 * 0.622 * (Ew/(p-Ew))/ 100.0
    return mixing_ratio

def calculate_rh2(mixing_ratio, qs):
    relative_humidity = (100 * mixing_ratio / qs)
    return relative_humidity

def calculate_water_year(data, method):
    year_list = []
    yearly_values = []
    for year in data.resample(time='y').sum().time.dt.year.values:
        time_start = str(year -1) + '-10-01'
        time_end = str(year) + '-09-30'
        if (time_start >= str(data.time[0].values)) and (time_end <= str(data.time[-1].values)):
            year_value = data.sel(time=slice(time_start, time_end))
            year_list.append(year)
            if method == 'sum':
                yearly_values.append(np.sum(year_value.values))
            elif method == 'mean':
                yearly_values.append(np.mean(year_value.values))
    return np.array(year_list), np.array(yearly_values)


def calculate_season(data, method, start, end, new_year=True):
    year_list = []
    seasonal_values = []
    for year in data.resample(time='y').sum().time.dt.year.values:
        time_start = str(year) + '-' + start
        if new_year == True:
            time_end = str(year+1) + '-' + end
        elif new_year == False:
            time_end = str(year) + '-' + end
        if (time_start >= str(data.time[0].values)) and (time_end <= str(data.time[-1].values)):
            year_value = data.sel(time=slice(time_start, time_end))
            year_list.append(year)
            if method == 'sum':
                seasonal_values.append(np.sum(year_value.values))
            elif method == 'mean':
                seasonal_values.append(np.mean(year_value.values))
    return np.array(year_list), np.array(seasonal_values)


# orig? def calculate_seasonal_characteristics(mb_ref, temp_positive, temp_negative, preci_positive, preci_negative):
def calculate_seasonal_characteristics(mb_ref, temp_positive, preci_positive):
    #breakpoint()
    temp_deviations = temp_positive - mb_ref
    preci_deviations = preci_positive - mb_ref
    return temp_deviations, preci_deviations

def calcualte_long_term_monthly_mean(cosi):
    ltm = []
    for i in range(1,13,1):
        ltm.append(cosi.MB.values[cosi.time.dt.month == i].mean())
    return ltm
