import numpy as np
import matplotlib as mpl; mpl.rcParams['figure.dpi'] = 300
import matplotlib.pyplot as plt
from scipy.stats import linregress, norm
from statsmodels.tools.eval_measures import rmse
from misc_functions.aggregation_functions import var_mean_time_total, var_sum_time_total, var_mean_space
from misc_functions.calculate_parameters import calculate_water_year
extension = '.png'

def plot_bar_monthly(var, plt_dir, file_name=None, temperature = 'K', barWidth = 0.4, mean_title=False, method='sum'):
    if temperature is 'C':
        var = var - 273.16
        ylabel='\u00B0C'
    if method == 'sum':
      monthly_values = var.resample(time='m', keep_attrs = True).sum(dim='time', keep_attrs = True)
    elif method == 'sum':
      monthly_values = var.resample(time='m', keep_attrs = True).mean(dim='time', keep_attrs = True)
#    breakpoint()
    plot_bar(monthly_values, var.long_name, '', var.units, plt_dir, 'monthly', barWidth=15)

def plot_line(var1,long_name, xlabel,ylabel,plt_dir,mean,temperature='K',save='False', cumulative=False, name=False, ylimmin=None, ylimmax=None):
    if temperature == 'K':
        var1_plot= var1
    elif temperature == 'C':
        var1_plot = var1 - 273.16
        ylabel='\u00B0C'
    plt.figure(figsize=(16, 9))
    var1_plot.plot(linewidth=3)
    #var1_plot.plot()
    plt.autoscale(enable=True, axis='y')
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=30, fontsize=20)
    plt.axhline(linewidth=3, color='b')
    plt.xlabel('')
    plt.grid(True)
    title = 'max, min, mean: ' + str(round(np.nanmax(var1_plot),2)) + ' ' + str(round(np.nanmin(var1_plot),2)) + ' ' + str(round(np.nanmean(var1_plot),2))
    plt.title(title)
    if cumulative == True:
        cumulative_mass_balance = float(var1_plot.values[-1])
        plt.title('Cumulative mass balance: ' + str(float(var1_plot.values[-1]))[0:6])
    if name == False:
        file_name = long_name.replace(" ", "_")+'-' + mean
    else:
        file_name = name + '_' + long_name.replace(" ", "_")+'-'+mean
    #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    plt_file = plt_dir + file_name + extension
    if save == False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()

def plot_line_water_year(var, plt_dir, temperature = 'K'):
    x, y, = calculate_water_year(var, 'mean')
    if temperature == 'C':
        y -= 273.16
        ylabel='\u00B0C'
    plt.figure(figsize=(16, 9))
    plt.plot(x,y,linewidth=3)
    plt.autoscale(enable=True, axis='y')
    plt.ylabel(var.units, fontsize=20)
    plt.xlabel('',fontsize=20 )
    plt.yticks(fontsize=20)
    plt.xticks(rotation=30, fontsize=20)
    plt.axhline(linewidth=3, color='b')
    plt.xlabel('')
    plt.grid(True)
    plt.title('')
    file_name = var.long_name.replace(" ", "_")
    plt_file = plt_dir + file_name + extension
    print("save plt")
    plt.savefig(plt_file)
    plt.close()

def plot_line_diff(var1, var2, label1, label2, long_name, xlabel, ylabel, plt_dir, supp_filename, temperature='K', albedo=False, totalheight=False ,diff='False',save='True', name=False, ylimmin=None, ylimmax=None):
    if temperature == 'K':
        var1_plot= var1
        var2_plot= var2
    elif temperature == 'C':
        var1_plot = var1 - 273.16
        var2_plot = var2 - 273.16
        ylabel = 'Surface temperature (\u00B0C)'
    plt.figure(figsize=(16, 9))
    var3_plot = var2_plot - var1_plot
    if totalheight == True:
        var1_plot.plot(label=label1,linewidth=3,color='k')
        var2_plot.plot(label=label2,linewidth=3,color='r')
        if diff == True:
            var3_plot.plot(label='Differenz\n ('+str(label2) + ' - ' + str(label1)+')',linewidth=3)
    else:
        var1_plot.plot(label=label1,color='k')
        var2_plot.plot(label=label2,color='r')
        if diff == True:
            var3_plot.plot(label='Differenz\n ('+str(label2) + ' - ' + str(label1)+')')
    plt.autoscale(enable=True, axis='y')
    if ylimmin != None:
        plt.ylim(ylimmin,ylimmax)
    plt.ylabel(ylabel, fontsize=20, labelpad=40)
    plt.xticks(rotation=30, fontsize=20)
    plt.xlabel('', fontsize=20)
    plt.yticks(fontsize=20)
#    plt.legend(loc="upper left", bbox_to_anchor=(0.80,1.15),fontsize=18)
    plt.legend(fontsize = 20)
    plt.grid(True)
    title = 'max, min, mean: ' + str(round(np.nanmax(var1_plot),2)) + ' ' + str(round(np.nanmin(var1_plot),2)) + ' ' + str(round(np.nanmean(var1_plot),2))
    plt.title(title, fontsize = 20)
    if name == False:
        file_name = 'timeline_' + supp_filename + '_' + long_name.replace(" ", "_")
    else:
        file_name = 'timeline_' + supp_filename + '-' + long_name.replace(" ", "_") + '_' + name
    #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    plt_file = plt_dir + file_name + '_comparison' + extension
    if save == False:
        plt.show()
    else:
        print("save plt")
        plt.savefig(plt_file)
        print(plt_file)
        plt.close()

def plot_line_3var(var1, var2, var3, label1, label2, label3, long_name, xlabel, ylabel, plt_dir, supp_filename, temperature='K', albedo=False, totalheight=False ,diff='False',save='True', name=False, ylimmin=None, ylimmax=None):
    if temperature == 'K':
        var1_plot = var1
        var2_plot = var2
        var3_plot = var3
    elif temperature == 'C':
        var1_plot = var1 - 273.16
        var2_plot = var2 - 273.16
        var3_plot = var3 - 273.16
        ylabel= '\u00B0C'
    plt.figure(figsize=(16, 9))
    if totalheight == True:
        var1_plot.plot(label=label1,linewidth=3,color='r')
        var2_plot.plot(label=label2,linewidth=3,color='g')
        var3_plot.plot(label=label3, linewidth=3, color='b')
    else:
        var1_plot.plot(label=label1, linewidth=3, color='r')
        var2_plot.plot(label=label2, linewidth=3, color='g')
        var3_plot.plot(label=label3, linewidth=3, color = 'b')

    plt.autoscale(enable=True, axis='y')
    if ylimmin != None:
        plt.ylim(ylimmin,ylimmax)
    plt.ylabel(ylabel, fontsize=20, labelpad=40)
    plt.xticks(rotation=30, fontsize=20)
    plt.xlabel('', fontsize=20)
    plt.yticks(fontsize=20)
    plt.title('')
    plt.legend(fontsize = 20)
    plt.grid(True)
    if name == False:
        file_name = 'timeline_' + supp_filename + '_' + long_name.replace(" ", "_")
    else:
        file_name = 'timelien_' + supp_filename + '-' + long_name.replace(" ", "_") + '_' + name
    #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    plt_file = plt_dir + file_name + '_comparison' + extension
    if save == False:
        plt.show()
    else:
        print("save plt")
        plt.savefig(plt_file)
        print(plt_file)
        plt.close()

def plot_bar(var1,long_name, xlabel,ylabel,plt_dir,mean,temperature='K',barWidth=1):
    if temperature == 'K':
        var1_plot = var1
    elif temperature == 'C':
        var1_plot = var1 - 273.16
        ylabel='\u00B0C'
    plt.figure(figsize=(16, 9))
    if mean == 'y':
        x = var1_plot.time.dt.year.values
        y = var1_plot
        barWidth = 0.5
    elif mean == 'm':
        x = var1_plot.time.dt.strftime('%Y %m').values
        y = var1_plot
        barWidth = 0.5
    else:
        x=var1_plot.time.values[0:-1]   ### otherwise 2001 for example is printed as 2002
        y=var1_plot[1:]                 ### otherwise 2001 for example is printed as 2002
    plt.bar(x,y, width=barWidth, align='center')
    plt.autoscale(enable=True, axis='y')
    #plt.grid(True)
    plt.ylabel(ylabel, fontsize=20)
    plt.xlabel(xlabel, fontsize=20)
    if len(x) > 10:
        plt.xticks(np.arange(0,len(x),round(len(x)/10)), rotation=30, fontsize=20)
    else:
        plt.xticks(rotation=30, fontsize=20)
    plt.yticks(fontsize=20)
    plt.axhline(linewidth=2, color='b')
    plt.title('')
    file_name = long_name.replace(" ", "_")+'-'+mean
    plt_file = plt_dir + 'BAR_plot_' + file_name + extension
    plt.savefig(plt_file)
    plt.close()
def plot_bar_water_year(var, plt_dir, file_name=None, temperature = 'K', barWidth = 0.4, mean_title=False):
    x, y, = calculate_water_year(var, 'sum')
    if temperature is 'C':
        y = y - 273.16
        ylabel='\u00B0C'
    plt.figure(figsize=(16, 9))
    plt.bar(x,y, width=barWidth, align='center')
    plt.autoscale(enable=True, axis='y')
    #plt.grid(True)
    plt.ylabel(var.units, fontsize=20,  labelpad=20)
    plt.grid()
    #plt.xlabel('', fontsize=20)
    #if len(x) > 10:
    #    plt.xticks(np.arange(0,len(x),round(len(x)/10)), rotation=30, fontsize=20)
    #    print(len(x))
    #else:
    plt.xticks(rotation=30, fontsize=20)
    plt.yticks(fontsize=20)
    plt.axhline(linewidth=2, color='b')
    if mean_title == False:
        plt.title('')
    else:
        title = 'Mean annual: ' +  str(round(np.nanmean(y), 2)) + ' ' + str(var.units)
        plt.title(title, fontsize=20)
    if file_name !=None:
      file_name = file_name + '_' + var.long_name.replace(" ", "_") + '_BAR_plot'
    else: 
      file_name = var.long_name.replace(" ", "_") + '_BAR_plot'

    plt_file = plt_dir + file_name 
    plt.savefig(plt_file)
    plt.close()

def plot_2var_bar(var1,var2,label1,label2,long_name, xlabel,ylabel,plt_dir,mean,temperature='K',save='True'):
    if temperature == 'K':
        var1_plot= var1
        var2_plot= var2
    elif temperature == 'C':
        var1_plot = var1 - 273.16
        var2_plot = var2 - 273.16
        ylabel='\u00B0C'
    barWidth = 100
    plt.figure(figsize=(16, 9))
    print("Does not work, addapted from normal BAR plot")
    # plt.bar(var1_plot.time.values, var1_plot, width=-barWidth ,align='edge', label=label1, color='b')
    # plt.bar(var1_plot.time.values, var2_plot, width=barWidth, align='center',label=label2, color='r')
    # plt.ylabel(ylabel, fontsize=20)
    # # plt.ylim(-10,2)
    # plt.xlabel(xlabel, fontsize=20)
    # plt.xticks(rotation=30, fontsize=20)
    # plt.yticks(fontsize=20)
    # #plt.autoscale(enable=True,axis='y')
    # plt.grid(True)
    # plt.legend(loc="upper left", bbox_to_anchor=(0.77,1.17),fontsize=20)
    # #plt.title(long_name)
    # file_name = long_name.replace(" ", "_")+'-bar-'+mean
    # #timestr = time.strftime("%Y%m%d") #("%Y%m%d-%H%M%S")
    # plt_file = plt_dir + 'comparison-' + file_name + extension
    # if save is False:
    #     plt.show()
    # else:
    #     print("save plt")
    #     plt.savefig(plt_file)
    #     plt.close()

def scatterplot_linear(var1, var2, xlabel, ylabel, long_name, mean, plt_dir, supp_filename, unit1, unit2, line_best_fit = True, temperature='K', name=False):
    mask = ~np.isnan(var1) & ~np.isnan(var2)
    var1 = var1[mask]
    var2 = var2[mask]
    if temperature == 'C':
        var1 = var1 - 273.16
        var2 = var2 - 273.16
        unit1 = '\u00B0C'
        unit2 = '\u00B0C'
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
    if line_best_fit == True:
        plt.plot(var1,y_pred,color='r', linewidth=5)
    plt.grid()
    plt.xlabel(xlabel +' '+unit1, fontsize=20, labelpad=10)
    plt.ylabel(ylabel +' '+unit2,  fontsize=20, labelpad=10)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    axis_min = min(np.nanmin(var1),np.nanmin(var2))
    axis_max = max(np.nanmax(var1),np.nanmax(var2))
    plt.ylim(axis_min, axis_max)
    plt.xlim(axis_min, axis_max)
    plt.plot([axis_min, axis_max],[axis_min, axis_max],linestyle='dashed', color='k')
    if R_squared < 0.01:
      r2_for_title = 0.00
    else:
      r2_for_title = R_squared.values
    print('r2:', R_squared.values)
    plt.title('R\xb2: ' +  str(np.round(R_squared.values, 2)) + ' RMSE: '+str(root_mean)[0:4], fontsize=20)
    if name == False:
        file_name = 'scatter_' + supp_filename + '_' + long_name.replace(" ", "_")
    else:
        file_name = 'scatter_' + supp_filename + '_' + long_name.replace(" ", "_")+'_' + name
    plt_file = plt_dir + file_name + extension
    plt.savefig(plt_file)
    plt.close()

def plot_pdf_3var(var1, var2, var3, xlabel, long_name, plt_dir, name=False, save=True):
    x1 = np.sort(var1)
    x2 = np.sort(var2)
    x3 = np.sort(var3)
    plt.figure(figsize=(16, 9))
    y1_pdf = norm.pdf(x1, loc=np.mean(x1), scale=np.std(x1))
    y2_pdf = norm.pdf(x2, loc=np.mean(x2), scale=np.std(x2))
    y3_pdf = norm.pdf(x3, loc=np.mean(x3), scale=np.std(x3))
    plt.plot(x1, y1_pdf, marker='v', markersize = '5', linestyle='none', label='measured values', color='r')
    plt.plot(x2, y2_pdf, marker='^', markersize = '5', linestyle='none', label='simulated values', color='g')
    plt.plot(x3, y3_pdf, marker='o', markersize = '3', linestyle='none', label='mapped values', color='b')
    plt.autoscale(enable=True, axis='y')
    #plt.ylabel('%', fontsize=20)
    plt.xlabel(long_name + ' (' + xlabel + ')', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.title('')
    if name == False:
        file_name = 'pdf_3var_' + long_name.replace(" ", "_")
    else:
        file_name = 'pdf_3var_' + name + '_' + long_name.replace(" ", "_")
    plt_file = plt_dir + file_name + extension
    if save == False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()

def plot_cdf(var, xlabel, long_name, plt_dir, name=False, save=True):
    x = np.sort(var)
    y = np.arange(1, len(x)+1)/len(x)
    plt.figure(figsize=(16, 9))
    plt.plot(x,y,marker='.', linestyle='none')
    plt.autoscale(enable=True, axis='y')
    #plt.ylabel('%', fontsize=20)
    plt.xlabel(long_name + ' (' + xlabel + ')', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.grid(True)
    plt.title('')
    if name == False:
        file_name = 'cdf_' + long_name.replace(" ", "_")
    else:
        file_name = 'cdf_' + name + '_' + long_name.replace(" ", "_")
    plt_file = plt_dir + file_name + extension
    if save == False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()

def plot_cdf_compare_2var(var1, var2, xlabel, long_name, plt_dir, name=False, save=True):
    x1 = np.sort(var1)
    x2 = np.sort(var2)
    y = np.arange(1, len(x1)+1)/len(x1)
    y2 = np.arange(1, len(x2) + 1) / len(x2)
    plt.figure(figsize=(16, 9))
    plt.plot(x1, y,marker='v', markersize = '5', linestyle='none', label='measured values', color='r')
    plt.plot(x2, y2, marker='^', markersize = '5', linestyle='none', label='simulated values', color='g')
    plt.autoscale(enable=True, axis='y')
    #plt.ylabel('%', fontsize=20)
    plt.xlabel(long_name + ' (' + xlabel + ')', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.title('')
    if name == False:
        file_name = 'cdf_2var_' + long_name.replace(" ", "_")
    else:
        file_name = 'cdf_2var_' + name + '_' + long_name.replace(" ", "_")
    plt_file = plt_dir + file_name + extension
    if save == False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()

def plot_cdf_compare_3var(var1, var2, var3, xlabel, long_name, label1, label2, label3, plt_dir, name=False, save=True, temperature='K'):
    if temperature == 'C':
        var1 -= 273.16
        var2 -= 273.16
        var3 -= 273.16
    x1 = np.sort(var1)
    x2 = np.sort(var2)
    x3 = np.sort(var3)
    y1 = np.arange(1, len(x1) + 1) / len(x1)
    y2 = np.arange(1, len(x2) + 1) / len(x2)
    y3 = np.arange(1, len(x3) + 1) / len(x3)
    plt.figure(figsize=(16, 9))
    plt.plot(x1, y1, linewidth=3, label=label1, color='r')
    plt.plot(x2, y2, linewidth=3, label=label2, color='g')
    plt.plot(x3, y3, linewidth=3, label=label3, color='b')
    plt.autoscale(enable=True, axis='y')
    #plt.ylabel('%', fontsize=20)
    plt.xlabel(xlabel, fontsize=20, labelpad=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.title('')
    if name == False:
        file_name = 'cdf_compare_3var_' + long_name.replace(" ", "_")
    else:
        file_name = 'cdf_compare_3var_' + name + '_' + long_name.replace(" ", "_")
    plt_file = plt_dir + file_name + extension
    if save == False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()

def plot_cdf_compare_4var(var1, var2, var3, var4, xlabel, long_name, plt_dir, name=False, save=True):
    x1 = np.sort(var1)
    x2 = np.sort(var2)
    x3 = np.sort(var3)
    x4 = np.sort(var4)
    y = np.arange(1, len(x1)+1)/len(x1)
    y4 = np.arange(1, len(x4) + 1)/len(x4)
    plt.figure(figsize=(16, 9))
    plt.plot(x1, y,marker='v', markersize = '5', linestyle='none', label='measured values', color='r')
    plt.plot(x2, y, marker='^', markersize = '5', linestyle='none', label='simulated values', color='g')
    plt.plot(x3, y, marker='o', markersize = '3', linestyle='none', label='mapped values', color='b')
    plt.plot(x4, y4, marker='o', markersize='3', linestyle='none', label='mapped values all', color='k')
    plt.autoscale(enable=True, axis='y')
    #plt.ylabel('%', fontsize=20)
    plt.xlabel(long_name + ' (' + xlabel + ')', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.title('')
    if name == False:
        file_name = 'cdf_compare_4var_' + long_name.replace(" ", "_")
    else:
        file_name = 'cdf_compare_4var_' + name + '_' + long_name.replace(" ", "_")
    plt_file = plt_dir + file_name + extension
    if save == False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()

def plot_cdf_compare_5var(var1, var2, var3, var4, var5, xlabel, long_name, plt_dir, name=False, save=True):
    x1 = np.sort(var1)
    x2 = np.sort(var2)
    x3 = np.sort(var3)
    x4 = np.sort(var4)
    x5 = np.sort(var5)
    y = np.arange(1, len(x1)+1)/len(x1)
    y4 = np.arange(1, len(x4) + 1)/len(x4)
    plt.figure(figsize=(16, 9))
    plt.plot(x1, y,marker='v', markersize = '5', linestyle='none', label='measured values', color='r')
    plt.plot(x2, y, marker='^', markersize = '5', linestyle='none', label='simulated values', color='g')
    plt.plot(x3, y, marker='o', markersize = '3', linestyle='none', label='mapped values', color='b')
    plt.plot(x4, y4, marker='o', markersize='3', linestyle='none', label='mapped values all', color='k')
    plt.plot(x5, y4, marker='o', markersize='3', linestyle='none', label='lapse rate values all', color='y')
    plt.autoscale(enable=True, axis='y')
    #plt.ylabel('%', fontsize=20)
    plt.xlabel(long_name + ' (' + xlabel + ')', fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True)
    plt.title('')
    if name == False:
        file_name = 'cdf_compare_5var_' + long_name.replace(" ", "_")
    else:
        file_name = 'cdf_compare_5var_' + name + '_' + long_name.replace(" ", "_")
    plt_file = plt_dir + file_name + extension
    if save == False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()

def plot_cumulative(var1,plt_dir,mean='h',save='False', name=False):
    var1_cum=np.cumsum(var1)
    print(var1_cum[-1])
    plot_line(var1_cum, 'cumulative_' + str(var1.long_name), mean, var1.units, plt_dir, mean, temperature='K',save='False', cumulative=True, name=name)


def plot_cumulative_terms(cosi, plt_dir, save=True):
    MB_mean = var_mean_space(cosi.MB)
    surfMB_mean = var_mean_space(cosi.surfMB)
    intMB_mean = var_mean_space(cosi.intMB)
    MB_cum = np.cumsum(MB_mean)
    surfMB_cum = np.cumsum(surfMB_mean)
    intMB_cum = np.cumsum(intMB_mean)
    plt.figure(figsize=(16, 9))
    MB_cum.plot(linewidth=10, label='MB')
    surfMB_cum.plot(linewidth=3, label='surface MB')
    intMB_cum.plot(linewidth=3, label='internal MB', color='k')
    plt.autoscale(enable=True, axis='y')
    plt.ylabel(cosi.MB.units, fontsize=20)
    plt.yticks(fontsize=20)
    plt.xticks(rotation=30, fontsize=20)
    #plt.axhline(linewidth=3, color='b')
    plt.xlabel('')
    plt.grid(True)
    plt.title('')
    file_name = 'Cumulative_mass_balance_terms'
    plt_file = plt_dir + file_name + extension
    plt.legend(fontsize=20)
    if save == False:
        plt.show()
        plt.close()
    else:
        print("save plt")
        plt.savefig(plt_file)
        plt.close()


def plot_cummulative_compare(var1,var2,label1,label2,content,plt_dir,mean='h'):
    var1_cum=np.cumsum(var1)
    var2_cum=np.cumsum(var2)
    plot_line_diff(var1_cum, var2_cum, label1, label2, 'cummulative_MB', mean, var1.units, plt_dir, mean, temperature='K')

def spatial_plot(var, long_name, plt_dir, name, temperature='K', save='True', vmin=None, vmax=None, discrete = True):
    plt.figure(figsize=(16, 9))
    if temperature == 'K':
        var_plot = var
    elif temperature == 'C':
        var_plot = var - 273.16
        var_plot.attrs['units'] = '\u00B0C'
    if np.max(var) - np.min(var) <= 0.000000001:
        cmap = 'Reds'
    else:
        cmap = 'RdBu_r'
    if (np.max(var) - np.min(var) >= 0.000000001) and (discrete == True):
        if vmin == None:
            levels = np.arange(np.min(var_plot), np.max(var_plot), ((np.max(var_plot) - np.min(var_plot)) / 20))
        else:
            levels = np.arange(vmin, vmax, (vmax - vmin) / 20)
        var_plot.plot.pcolormesh(cmap=cmap, levels=levels, vmin=vmin, vmax=vmax)
    else:
        var_plot.plot.pcolormesh(cmap=cmap)
    plt.title(long_name)
    if name != None:
        file_name = long_name.replace(" ", "_")+'-'+ name
    else:
        file_name = long_name.replace(" ", "_")
    plt_path = plt_dir + file_name + extension
    if save == False:
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
            if var.name == 'HGT' or var.name == 'REFERENCEHEIGHT' or var.name == 'Qcum' or var.name =='MASK' or var.name == 'ASPECT' or var.name == 'SLOPE':
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
                cosi[var.name].values[:,cosi.MASK.values!=1] = np.nan
                var_aggregate_space = var_mean_space(cosi[var.name])
                plot_line(var_aggregate_space.resample(time=mean).sum(), var.long_name,mean,var.units,plt_dir,mean)
                print("1st", var.units)

            elif var.units == 'W m\u207b\xb2' or var.units == 'm' or var.units == '%' or var.units == 'm s\u207b\xb9' or var.units == 'hPa' or var.units == '-':
                cosi[var.name].values[:,cosi.MASK.values!=1] = np.nan
                var_aggregate_space = var_mean_space(cosi[var.name])
                print(var.name)
                plot_line(var_aggregate_space.resample(time=mean).mean(), var.long_name,mean,var.units,plt_dir,mean)
                print("2st", var.units)

            elif var.units == 'K':
                cosi[var.name].values[:,cosi.MASK.values!=1] = np.nan
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
            if var.name == 'HGT' or var.name == 'MASK' or var.name == 'ASPECT' or var.name == 'SLOPE':
                print('Plot static variable: %s' %(var.name))
                cosi[var.name].values = cosi[var.name].values.astype(float)
                cosi[var.name].values[cosi.MASK.values!=1]=np.nan
                spatial_plot(cosi[var.name], var.name, plt_dir, name)

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

def plot_all_spatial(cosi,plt_dir,name=None, save=True):
    for varname, var in cosi.data_vars.items():
        print(var.name)
        if var.isnull().all():
            print(var.name,"nothing to plot")
        else:
            spatial_plot(cosi[var.name], var.long_name, plt_dir, name)

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

def plot_water_year_dataset(dataset,plt_dir):
    plot_bar_water_year(dataset.MB, plt_dir)
    plot_bar_water_year(dataset.RRR, plt_dir)
    plot_bar_water_year(dataset.SNOWFALL, plt_dir)
    plot_line_water_year(dataset.T2, plt_dir, temperature = 'C')
    plot_line_water_year(dataset.RH2, plt_dir)
    plot_line_water_year(dataset.U2, plt_dir)
    plot_line_water_year(dataset.PRES, plt_dir)
    plot_line_water_year(dataset.G, plt_dir)
    plot_line_water_year(dataset.N, plt_dir)
    plot_line_water_year(dataset.LWin, plt_dir)


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

            elif var.units == 'K':
                plot_line(cosi[var.name].resample(time=mean).mean(), var.long_name,mean,'\u00B0C',plt_dir,mean,'C')
                print("3st", var.units)
            else:
                print(var.units,"variable not marked as plotfind variable: ", var.name)

#def plot_variables_water_year(dataset, plt_dir):

