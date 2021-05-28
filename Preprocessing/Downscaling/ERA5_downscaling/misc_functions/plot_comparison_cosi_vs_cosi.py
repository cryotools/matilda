from functions.plot_functions import *

def plot_all_compare(cosi1,cosi2,label1,label2,content,plt_dir,mean='h'):
    for varname, var in cosi1.data_vars.items():
        if var.isnull().all():
            print(var.name,"nothing to plot")
        else:
            if var.name == 'HGT' or var.name == 'REFERENCEHEIGHT' or var.name == 'Qcum':
                print("Static variable or only COSIMA VARIABLE: ",var.name)

            elif var.name == 'T2' or var.name == 'RH2' or var.name == 'U2' or var.name == 'RAIN' or var.name == 'SNOWFALL' \
                    or var.name == 'PRES' or var.name == 'N' or var.name == 'G':
                print("Input variable, not interesting at the moment: ", var.name)

            elif var.name == 'LAYER_HEIGT' or var.name == 'LAYER_RHO' or var.name == 'LAYER_T' or var.name == 'LAYER_LWC' \
                    or var.name == 'LAYER_CC' or var.name == 'LAYER_POROSITY' or var.name == 'LAZER_VOL' \
                    or var.name == 'LAYER_REFREEZE':
                print("4D variable, not interesting at the moment: ", var.name)

            elif var.name == 'T2' or var.name == 'RH2' or var.name == 'U2' or var.name == 'RAIN' or var.name == 'SNOWFALL' \
                    or var.name == 'PRES' or var.name == 'N' or var.name == 'G':
                print("Input variable not interesting at the moment: ", var.name)

            elif var.units == 'm w.e.' or var.units == 'mm' or var.name == 'SNOWFALL':
                plot_line_diff(cosi1[var.name].resample(time=mean).sum(), cosi2[var.name].resample(time=mean).sum(), label1,label2,
                               var.long_name,mean,var.units,plt_dir,mean)
                print("1st", var.units)
            elif var.units == 'W m\u207b\xb2' or var.units == 'm' or var.units == '%' or var.units == 'm s^-1' or var.units == 'hPa' or var.units == '-':
                plot_line_diff(cosi1[var.name].resample(time=mean).mean(), cosi2[var.name].resample(time=mean).mean(),label1,label2,
                               var.long_name,mean,var.units,plt_dir,mean)
                print("2st", var.units)

            elif var.units is 'K':
                plot_line_diff(cosi1[var.name].resample(time=mean).mean(), cosi2[var.name].resample(time=mean).mean(),label1,label2,
                               var.long_name,mean,'\u00B0C',plt_dir,mean,'C')
                print("3st", var.units)
            else:
                print(var.units,"variable not marked as plot find variable: ", var.name)





def plot_all_compare_bar(cosi1,cosi2,label1,label2,content,plt_dir,mean='h'):
    for varname, var in cosi1.data_vars.items():
        if var.isnull().all():
            print(var.name,"nothing to plot")
        else:
            if var.name == 'HGT' or var.name == 'REFERENCEHEIGHT' or var.name == 'Qcum':
                print("Static variable or only COSIMA VARIABLE: ",var.name)

            elif var.name == 'T2' or var.name == 'RH2' or var.name == 'U2' or var.name == 'RAIN' or var.name == 'SNOWFALL' \
                    or var.name == 'PRES' or var.name == 'N' or var.name == 'G':
                print("Input variable, not interesting at the moment: ", var.name)

            elif var.name == 'LAYER_HEIGT' or var.name == 'LAYER_RHO' or var.name == 'LAYER_T' or var.name == 'LAYER_LWC' \
                    or var.name == 'LAYER_CC' or var.name == 'LAYER_POROSITY' or var.name == 'LAZER_VOL' \
                    or var.name == 'LAYER_REFREEZE':
                print("4D variable, not interesting at the moment: ", var.name)

            elif var.name == 'T2' or var.name == 'RH2' or var.name == 'U2' or var.name == 'RAIN' or var.name == 'SNOWFALL' \
                    or var.name == 'PRES' or var.name == 'N' or var.name == 'G':
                print("Input variable not interesting at the moment: ", var.name)

            elif var.units == 'm w.e.' or var.units == 'mm' or var.name == 'SNOWFALL':

                plot_2var_bar(cosi1[var.name].resample(time=mean).sum(), cosi2[var.name].resample(time=mean).sum(), label1, label2,
                              cosi1[var.name].long_name, mean, cosi1[var.name].units, plt_dir, mean)
                print("1st", var.units)
            elif var.units == 'W m\u207b\xb2' or var.units == 'm' or var.units == '%' or var.units == 'm s^-1' or var.units == 'hPa' or var.units == '-':
                plot_2var_bar(cosi1[var.name].resample(time=mean).mean(), cosi2[var.name].resample(time=mean).mean(),
                              label1, label2,
                              cosi1[var.name].long_name, mean, cosi1[var.name].units, plt_dir, mean)

                print("2st", var.units)

            elif var.units is 'K':
                plot_2var_bar(cosi1[var.name].resample(time=mean).mean(), cosi2[var.name].resample(time=mean).mean(),
                              label1, label2,
                              cosi1[var.name].long_name, mean, cosi1[var.name].units, plt_dir, mean)
                print("3st", var.units)
            else:
                print(var.units,"variable not marked as plot find variable: ", var.name)
