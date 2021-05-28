import numpy as np; import sys
from scipy.stats import linregress
from statsmodels.tools.eval_measures import rmse

#################################
#nan functions ignore nan values#
#################################

### print max and min
def mm_nan(input): 
    print("max: ", np.nanmax(input))
    print("min: ", np.nanmin(input))
    print("")

### print max, min, mean, sum and number values
def mmm_nan(input):
    print("max: ", np.nanmax(input))
    print("min: ", np.nanmin(input))
    print("mean: ", np.nanmean(input))
    print("sum: ", np.nansum(input))
    print("number of values: ", input.shape)
    print("")

### print name, max, min, mean, sum and number values
def mmm_nan_name(input):
    print("variable: ", input.name)
    print("max: ", np.nanmax(input))
    print("min: ", np.nanmin(input))
    print("mean: ", np.nanmean(input))
    print("sum: ", np.nansum(input))
    print("number of values: ", input.shape)
    print("")

### print max, min, mean, sum and number values
def mmm(input):
    print("max: ", np.max(input))
    print("min: ", np.min(input))
    print("mean: ", np.mean(input))
    print("sum: ", np.sum(input), )
    print("number of values: ", input.shape)
    print("")

def mmm_nan_accumulated(input):
    print("max: ", np.nanmax(input))
    print("min: ", np.nanmin(input))

    print("total mean: ", np.nanmean(input))
    print("total sum: ", np.nansum(input))
    print("mean over gridpoints, sum over time: ", np.nansum(np.nanmean(input,axis=(0,1))))
    print("number of values: ", input.shape)
    print("number of values: ", input.shape)
    print("")

### print index where greater than and number ov falues 
def where_gt(input, arg):
    count = np.where(input > arg)
    quantity_condition= np.count_nonzero(count)
    print(count)
    print(quantity_condition)
    return count, quantity_condition

### print index where lower than and number ov falues 
def where_lt(input, arg):
    count = np.where(input < arg)
    quantity_condition= np.count_nonzero(count)
    print(count)
    print(quantity_condition)
    return count, quantity_condition

def calculate_r2_and_rmse(var1, var2, var):
    mask = ~np.isnan(var1) & ~np.isnan(var2)
    var1 = var1[mask]
    var2 = var2[mask]
    b, a, r, p, std = linregress(var1,var2)
    denominator = var1.dot(var1) - np.nanmean(var1) * np.nansum(var1)
    m = (var1.dot(var2) - np.nanmean(var2) * np.nansum(var1)) /denominator
    b = (np.nanmean(var2) * var1.dot(var1) - np.nanmean(var1) * var1.dot(var2)) /denominator
    y_pred = m*var1 + b
    res = var2 - y_pred
    tot = var2 - np.nanmean(var2)
    R_squared = 1 -  res.dot(res)/tot.dot(tot)
    root_mean = rmse(var1,var2)
    if R_squared >= 0.1:
        print('\n var: ', var, ' R2: ', R_squared.values, ' RMSE: ', root_mean, ' p-value: ', round(p,10), ' p-value: ', p)
    return R_squared, root_mean, p

### check if data greater or lower than max min
def check_reasonable(array,max,min):
    if np.nanmax(array) > max or np.nanmin(array) < min:
        print("CHECK your input DATA, the are out of a reasonable range!!!! ", str.capitalize(array.name)," MAX IS: ", np.nanmax(array), " AND MIN IS: ", np.nanmin(array))
    else:
        print(array.name, " max: ", np.nanmax(array), ", min: ", np.nanmin(array), " and mean: ", np.nanmean(array), )

def check(field, name, max, min):
    '''Check the validity of the input data '''
    if np.nanmax(field) > max or np.nanmin(field) < min:
        print('WARNING! Please check the data, its seems they are out of a reasonable range %s MAX: %.2f MIN: %.2f \n' % (str.capitalize(name), np.nanmax(field), np.nanmin(field)))

def check_for_nans_dataframe(dataframe):
    for col in dataframe.columns:
        if dataframe[col].isna().any():
            print('ERROR!!!!!!!!!!!: ', col, ' contains NaNs')
            sys.exit()

def check_for_nans_array(da):
    if np.isnan(da).any():
        print('ERRRRROR!!!: ', da.name, ' contains NaNs')
