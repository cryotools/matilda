import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from statsmodels.tools.eval_measures import rmse
from misc_functions.aggregation_functions import var_mean_time_total, var_sum_time_total, var_mean_space

def plot_line(var1,long_name, xlabel,ylabel,plt_dir,mean,temperature='K',save='False', cumulative=False, name=False, ylimmin=None, ylimmax=None):
    if temperature is 'K':
        var1_plot= var1
    elif temperature is 'C':
        var1_plot = var1 - 273.16
        ylabel='\u00B0C'
    plt.figure(figsize=(16, 9))
    var1_plot.plot(linewidth=3)
    #var1_plot.plot()
    plt.autoscale(enable=True, axis='y')
    plt.ylabel(ylabel, fontsize=20)
    if ylimmin is not None:
        plt.ylim(ylimmin,ylimmax)
    plt.xlabel(xlabel,fontsize=20 )
    plt.yticks(fontsize=20)
    plt.xticks(rotation=30, fontsize=20)
    plt.axhline(linewidth=3, color='b')
    plt.xlabel('')
    plt.grid(True)
    plt.title('')
    if cumulative is True:
        cumulative_mass_balance = float(var1_plot.values[-1])
        plt.title('Cumulative mass balance: ' + str(float(var1_plot.values[-1]))[0:6])
    if name is False:
        file_name = long_name.replace(" ", "_")+'-'+mean
    else:
        file_name = name + '_' + long_name.replace(" ", "_")+'-'+mean
    #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    plt_file = plt_dir + file_name + '.png'
    if save is False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()


def plot_line_diff(var1, var2, label1, label2, long_name, xlabel, ylabel, plt_dir, supp_filename, temperature='K', albedo=False, totalheight=False ,diff='False',save='True', name=False, ylimmin=None, ylimmax=None):
    if temperature is 'K':
        var1_plot= var1
        var2_plot= var2
    elif temperature is 'C':
        var1_plot = var1 - 273.16
        var2_plot = var2 - 273.16
        ylabel= ylabel + ' (' + '\u00B0C' + ')'
    plt.figure(figsize=(16, 9))
    var3_plot = var2_plot - var1_plot
    if totalheight is True:
        var1_plot.plot(label=label1,linewidth=3,color='k')
        var2_plot.plot(label=label2,linewidth=3,color='r')
        if diff is True:
            var3_plot.plot(label='Differenz\n ('+str(label2) + ' - ' + str(label1)+')',linewidth=3)
    else:
        var1_plot.plot(label=label1,color='k')
        var2_plot.plot(label=label2,color='r')
        if diff is True:
            var3_plot.plot(label='Differenz\n ('+str(label2) + ' - ' + str(label1)+')')
    plt.autoscale(enable=True, axis='y')
    if ylimmin is not None:
        plt.ylim(ylimmin,ylimmax)
    plt.ylabel(ylabel, fontsize=20, labelpad=40)
    plt.xticks(rotation=30, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.yticks(fontsize=20)
#    plt.legend(loc="upper left", bbox_to_anchor=(0.80,1.15),fontsize=18)
    plt.legend(fontsize = 20)
    plt.grid(True)
    plt.title(xlabel, fontsize = 20)
    if name is False:
        file_name = supp_filename + '-' + long_name.replace(" ", "_")
    else:
        file_name = supp_filename + '-' + long_name.replace(" ", "_")+'_' + name
    #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    plt_file = plt_dir + file_name + '_comparison.png'
    if save is False:
        plt.show()
    else:
        print("save plt")
        plt.savefig(plt_file)
        print(plt_file)
        plt.close()

def plot_bar(var1,long_name, xlabel,ylabel,plt_dir,mean,temperature='K'):
    if temperature is 'K':
        var1_plot= var1
    elif temperature is 'C':
        var1_plot = var1 - 273.16
        ylabel='\u00B0C'
    barWidth = 1
    plt.figure(figsize=(16, 9))
    x=var1_plot.time.values[0:-1]
    y=var1_plot[1:]
    plt.bar(x,y, width=barWidth, align='center')
    print("stop")
    #plt.bar(var1_plot.time.values, var1_plot, width=barWidth ,align='center')
    plt.autoscale(enable=True, axis='y')
    plt.grid(True)
    #plt.ylim(-10,2)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.xticks(rotation=30, fontsize=20)
    plt.yticks(fontsize=20)
    plt.axhline(linewidth=2, color='b')
    #plt.ylim(-0.9,0.9)
    plt.xlabel('')
    plt.title('')
    file_name = long_name.replace(" ", "_")+'-'+mean
    #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    plt_file = plt_dir + 'cosipy-' + file_name + '.png'
    plt.savefig(plt_file)
    plt.close()

def plot_2var_bar(var1,var2,label1,label2,long_name, xlabel,ylabel,plt_dir,mean,temperature='K',save='True'):
    if temperature is 'K':
        var1_plot= var1
        var2_plot= var2
    elif temperature is 'C':
        var1_plot = var1 - 273.16
        var2_plot = var2 - 273.16
        ylabel='\u00B0C'
    barWidth = 100
    plt.figure(figsize=(16, 9))
    plt.bar(var1_plot.time.values, var1_plot, width=-barWidth ,align='edge', label=label1, color='b')
    plt.bar(var1_plot.time.values, var2_plot, width=barWidth, align='center',label=label2, color='r')
    plt.ylabel(ylabel, fontsize=20)
    # plt.ylim(-10,2)
    plt.xlabel(xlabel, fontsize=20)
    plt.xticks(rotation=30, fontsize=20)
    plt.yticks(fontsize=20)
    #plt.autoscale(enable=True,axis='y')
    plt.grid(True)
    plt.legend(loc="upper left", bbox_to_anchor=(0.77,1.17),fontsize=20)
    #plt.title(long_name)
    file_name = long_name.replace(" ", "_")+'-bar-'+mean
    #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    plt_file = plt_dir + 'comparison-' + file_name + '.png'
    if save is False:
        plt.show()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()

def scatterplot_linear(var1, var2, xlabel, ylabel, long_name, mean, plt_dir, supp_filename, unit, line_best_fit = False, temperature='K', name=False):
    mask = ~np.isnan(var1) & ~np.isnan(var2)
    var1 = var1[mask]
    var2 = var2[mask]
    if temperature is 'K':
        var1 = var1
        var2 = var2
        unit = unit
    elif temperature is 'C':
        var1 = var1 - 273.16
        var2 = var2 - 273.16
        unit = '\u00B0C'
    #plt.figure(figsize=(16, 9))
    plt.figure(figsize=(12, 12))
    plt.scatter(var1,var2)
    b, a, r, p, std = linregress(var1,var2)
    denominator = var1.dot(var1) - np.nanmean(var1) * np.nansum(var1)
    m = (var1.dot(var2) - np.nanmean(var2) * np.nansum(var1)) /denominator
    b = (np.nanmean(var2) * var1.dot(var1) - np.nanmean(var1) * var1.dot(var2)) /denominator
    y_pred = m*var1 + b
    res = var2 - y_pred
    tot = var2 - np.nanmean(var2)
    R_squared = 1 -  res.dot(res)/tot.dot(tot)
    root_mean = rmse(var1,var2)
    if line_best_fit is True:
        plt.plot(var1,y_pred,'r')

    plt.xlabel(xlabel +' ('+unit+')', fontsize=20)
    plt.ylabel(ylabel +' ('+unit+')',  fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    #plt.ylim(-30,1)
    #plt.xlim(-30,1)

    plt.title('R\xb2: ' +  str(R_squared.values)[0:4] + ' RMSE: '+str(root_mean)[0:4], fontsize=20)
    if name is False:
        file_name = supp_filename + '-' + long_name.replace(" ", "_")
    else:
        file_name = supp_filename + '-' + long_name.replace(" ", "_")+'_' + name
    #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    plt_file = plt_dir + file_name + '_scatter.png'
    plt.savefig(plt_file)
    plt.close()

def plot_cumulative(var1,plt_dir,mean='h',save='False', name=False):
    var1_cum=np.cumsum(var1)
    print(var1_cum[-1])
    plot_line(var1_cum, 'cumulative_' + str(var1.long_name), mean, var1.units, plt_dir, mean, temperature='K',save='False', cumulative=True, name=name)

def plot_cummulative_compare(var1,var2,label1,label2,content,plt_dir,mean='h'):
    var1_cum=np.cumsum(var1)
    var2_cum=np.cumsum(var2)
    plot_line_diff(var1_cum, var2_cum, label1, label2, 'cummulative_MB', mean, var1.units, plt_dir, mean, temperature='K')

def spatial_plot(var, long_name, plt_dir, name, temperature='K', save='True'):
    plt.figure(figsize=(16, 9))
    if temperature is 'K':
        var_plot = var
    elif temperature is 'C':
        var_plot = var - 273.16
        var_plot.attrs['units'] = '\u00B0C'
    var_plot.plot.pcolormesh('lat', 'lon')
    plt.title(long_name)
    if name != None:
        file_name = long_name.replace(" ", "_")+'-'+ name
    else:
        file_name = long_name.replace(" ", "_")
    plt_path = plt_dir + file_name + '.png'
    if save is False:
        plt.show()
    else:
        print("save plt: ", plt_path)
        plt.savefig(plt_path)
        plt.close()

def plot_all_timeline(cosi,plt_dir,mean='h'):
    for varname, var in cosi.data_vars.items():
        if var.isnull().all():
            print(var.name,"nothing to plot")
        else:
            if var.name == 'HGT' or var.name == 'REFERENCEHEIGHT' or var.name == 'Qcum':
                print("Static variable or only COSIMA VARIABLE: ",var.name)

            # elif var.name == 'T2' or var.name == 'RH2' or var.name == 'U2' or var.name == 'RAIN' or var.name == 'SNOWFALL' \
            #         or var.name == 'PRES' or var.name == 'N' or var.name == 'G':
            #     print("Input variable, not interesting at the moment: ", var.name)

            elif var.name == 'LAYER_HEIGT' or var.name == 'LAYER_RHO' or var.name == 'LAYER_T' or var.name == 'LAYER_LWC' \
                    or var.name == 'LAYER_CC' or var.name == 'LAYER_POROSITY' or var.name == 'LAZER_VOL' \
                    or var.name == 'LAYER_REFREEZE':
                print("4D variable, plot: ", var.name)
                var_2_plot = cosi[var.name].mean(dim='layer',keep_attrs=True)
                var_aggregate_space = var_mean_space(var_2_plot)
                plot_line(var_aggregate_space.resample(time=mean).mean(), var.long_name,mean,var.units,plt_dir,mean)

            elif var.units == 'm w.e.' or var.units == 'mm' or var.name == 'SNOWFALL':
                var_aggregate_space = var_mean_space(cosi[var.name])
                plot_line(var_aggregate_space.resample(time=mean).sum(), var.long_name,mean,var.units,plt_dir,mean)
                print("1st", var.units)

            elif var.units == 'W m\u207b\xb2' or var.units == 'm' or var.units == '%' or var.units == 'm s\u207b\xb9' or var.units == 'hPa' or var.units == '-':
                var_aggregate_space = var_mean_space(cosi[var.name])
                plot_line(var_aggregate_space.resample(time=mean).mean(), var.long_name,mean,var.units,plt_dir,mean)
                print("2st", var.units)

            elif var.units is 'K':
                var_aggregate_space = var_mean_space(cosi[var.name])
                plot_line(var_aggregate_space.resample(time=mean).mean(), var.long_name,mean,'\u00B0C',plt_dir,mean,'C')
                print("3st", var.units)
            else:
                print(var.units,"variable not marked as plotfind variable: ", var.name)

def plot_interesting_spatial(cosi,plt_dir,name=None, save=True):
    for varname, var in cosi.data_vars.items():
        print(var.name)
        if var.isnull().all():
            print(var.name,"nothing to plot")
        else:
            if var.name == 'HGT':
                print('Plot static variable: %s' %(var.name))
                #cosi[var.name].values = cosi[var.name].values.astype(float)
                #cosi[var.name].values[cosi.MASK.values!=1]=np.nan
                spatial_plot(cosi[var.name], var.long_name, plt_dir, name)

            elif var.name == 'T2' or var.name == 'TS':
                print('Plot Celsius variables: %s' %(var.name))
                var_mean = var_mean_time_total(cosi[var.name])
                var_mean.values[cosi.MASK.values!=1]=np.nan
                spatial_plot(var_mean, var.long_name, plt_dir, name, temperature='C')

            elif var.name in {'RH2', 'U2', 'PRES', 'N', 'G', 'LWin', 
                    'LWout', 'H', 'LE', 'B', 'ME', 'SNOWHEIGHT', 'TOTALHEIGHT', 'Z0', 'ALBEDO',
                    'NLAYERS'}:
                print('Plot mean variable: %s' %(var.name))
                var_mean = var_mean_time_total(cosi[var.name])
                var_mean.values[cosi.MASK.values!=1]=np.nan
                spatial_plot(var_mean, var.long_name, plt_dir, name)

            elif var.name in {'RRR', 'RAIN', 'SNOWFALL', 'MB', 'surfMB', 'intMB', 'EVAPORATION', 'SUBLIMATION', 'CONDENSATION',
                    'DEPOSITION', 'surfMB', 'surfM', 'subM', 'Q', 'REFREEZE'}:
                print('Plot accumulated  variable: %s' %(var.name))
                var_mean = var_sum_time_total(cosi[var.name])
                var_mean.values[cosi.MASK.values!=1]=np.nan
                spatial_plot(var_mean, var.long_name, plt_dir, name)

            else:
                print('Do not know how to plot: ', var.name)


def plot_all_1D(cosi,plt_dir,mean='h', name=False):
    for varname, var in cosi.data_vars.items():
        if var.isnull().all():
            print(var.name,"nothing to plot")
        else:
            if var.name == 'HGT' or var.name == 'REFERENCEHEIGHT' or var.name == 'Qcum':
                print("Static variable or only COSIMA VARIABLE: ",var.name)

            # elif var.name == 'T2' or var.name == 'RH2' or var.name == 'U2' or var.name == 'RAIN' or var.name == 'SNOWFALL' \
            #         or var.name == 'PRES' or var.name == 'N' or var.name == 'G':
            #     print("Input variable, not interesting at the moment: ", var.name)

            elif var.name == 'LAYER_HEIGT' or var.name == 'LAYER_RHO' or var.name == 'LAYER_T' or var.name == 'LAYER_LWC' \
                    or var.name == 'LAYER_CC' or var.name == 'LAYER_POROSITY' or var.name == 'LAZER_VOL' \
                    or var.name == 'LAYER_REFREEZE':
                print("4D variable, plot: ", var.name)
                var_2_plot = cosi[var.name].mean(dim='layer',keep_attrs=True)
                plot_line(var_2_plot.resample(time=mean).mean(), var.long_name,mean,var.units,plt_dir,mean, name=name)

            elif var.units == 'm w.e.' or var.units == 'mm' or var.name == 'SNOWFALL':
                plot_line(cosi[var.name].resample(time=mean).sum(), var.long_name,mean,var.units,plt_dir,mean, name=name)
                print("1st", var.units)

            elif var.units == 'W m\u207b\xb2' or var.units == 'm' or var.units == '%' or var.units == 'm s\u207b\xb9' or var.units == 'hPa' or var.units == '-':
                plot_line(cosi[var.name].resample(time=mean).mean(), var.long_name,mean,var.units,plt_dir,mean, name=name)
                print("2st", var.units)

            elif var.units is 'K':
                plot_line(cosi[var.name].resample(time=mean).mean(), var.long_name,mean,'\u00B0C',plt_dir,mean,'C', name=name)
                print("3st", var.units)
            else:
                print(var.units,"variable not marked as plotfind variable: ", var.name)


def plot_Halji_precipitation_variables(cosi,plt_dir,mean='h'):
    for varname, var in cosi.data_vars.items():
        if var.isnull().all():
            print(var.name,"nothing to plot")
        else:
            if var.units == 'mm' or var.name == 'SNOWFALL' or var.units == 'mm/h':
                plot_line(cosi[var.name].resample(time=mean).sum(), var.long_name,mean,var.units,plt_dir,mean)
                print("1st", var.units)

            elif var.units == 'W m\u207b\xb2' or var.units == 'm' or var.units == '%' or var.units == 'm s\u207b\xb9' \
                    or var.units == 'hPa' or var.units == '-' or var.units == 'Zaehler':
                plot_line(cosi[var.name].resample(time=mean).mean(), var.long_name,mean,var.units,plt_dir,mean)
                print("2st", var.units)

            elif var.units is 'K':
                plot_line(cosi[var.name].resample(time=mean).mean(), var.long_name,mean,'\u00B0C',plt_dir,mean,'C')
                print("3st", var.units)
            else:
                print(var.units,"variable not marked as plotfind variable: ", var.name)
