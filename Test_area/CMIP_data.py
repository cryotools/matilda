"""
{Tool to iterate through CMIP6 ensemble outputs downloaded to 'CIRRUS' HPC at Humboldt-University.
Outputs various combinations of CSV timeseries.}

{Authors: Phillip Schuster & Ana-Lena Tappe}
"""

import xarray as xr
import pandas as pd
from pathlib import Path
import glob
import subprocess
import matplotlib.pyplot as plt
import os
home = str(Path.home())
#wd = home + "/Seafile/Tianshan_data/CMIP/CMIP6/dir_test"
#wd = "/home/phillip/Desktop/in_cm4_8/"
wd = "/data/projects/ensembles/cmip6"
output = "/data/projects/ebaca/Ana-Lena_Phillip/data/input_output/input/CMIP6/jyrgalang/"
## output information
var = ['near_surface_air_temperature', 'precipitation']
variable = ["tas", "pr"]
scen = ['historical', 'ssp1_2_6', 'ssp2_4_5', 'ssp3_7_0', 'ssp5_8_5']   #  'historical', 'ssp1_2_6', 'ssp2_4_5', 'ssp3_7_0', 'ssp5_8_5'
start_date = '1982-01-01'; end_date = '2100-12-31'
lat = 42.516; lon = 79.0167

plot = False
write_files = True

## find models: python version
def list_files(directory):
    paths = []
    subdirs = [x[0] for x in os.walk(directory)]
    for subdir in subdirs:
        files = os.walk(subdir).__next__()[2]
        if len(files) > 0:
            for file in files:
                paths.append(os.path.join(subdir, file))
    return paths


def list_models(wd, var, scen):
    models_scen = []
    if isinstance(var, list):
        if isinstance(scen, list):
            for v in var:
                for s in scen:
                    models = []
                    path = list_files(wd + '/' + v + '/' + s + '/')
                    for p in path:
                        mod = p.split(s)[1].split('/')
                        models.append(mod[len(mod) - 2])
                        models = list(set(models))
                    models_scen = models_scen + models
            models_scen = list(set([i for i in models_scen if models_scen.count(i) >= len(var)*len(scen)]))

        else:
            for v in var:
                models = []
                path = list_files(wd + '/' + v + '/' + scen + '/')
                for p in path:
                    mod = p.split(scen)[1].split('/')
                    models.append(mod[len(mod) - 2])
                    models = list(set(models))
                models_scen = models_scen + models
            models_scen = list(set([i for i in models_scen if models_scen.count(i) >= len(var)]))

    elif isinstance(scen, list):
        for s in scen:
            models = []
            path = list_files(wd + '/' + var + '/' + s + '/')
            for p in path:
                mod = p.split(s)[1].split('/')
                models.append(mod[len(mod) - 2])
                models = list(set(models))
            models_scen =  models_scen + models
        models_scen = list(set([i for i in models_scen if models_scen.count(i) >= len(scen)]))

    else:
        path = list_files(wd + '/' + var + '/' + scen + '/')
        models = []
        for p in path:
            mod = p.split(scen)[1].split('/')
            models.append(mod[len(mod) - 2])
        models_scen = list(set(models))

    return models_scen

model_list = list_models(wd, var, scen)
print(model_list)
if 'historical' in scen:
    print('\n' + 'A total of ' + str(len(model_list)) + ' models are available for the variable(s) ' +
          ' and '.join(var) + ' under the scenario(s) ' + ' and '.join(scen) + '.' + '\n')
else:
    print('\n' + 'A total of ' + str(len(model_list)) + ' models are available for the variable(s) ' +
          ' and '.join(var) + ' under the RCP(s) ' +
          ' and '.join([i.split('_')[1] + '.' + i.split('_')[2] for i in scen]) + '.' + '\n')

##
path = wd
scenarios = scen.copy()
var_id = variable.copy()
var_name = var.copy()

def cmip_csv(path, scenarios, model_list, var_id, var_name, lat, lon, start_date, end_date, method='nearest'):
    rcp_dfs = {}

    def cmip_open(path, lat, lon, var_id, method):
        ds = xr.open_dataset(path)
        pick = ds.sel(lat=lat, lon=lon, method=method)
        dat = pick[[var_id]].to_dataframe()
        if not isinstance(dat.index, pd.DatetimeIndex):
            #time = ds.indexes['time'].to_datetimeindex()
            time = ds.indexes['time']
            time = pd.to_datetime(time.astype('str'), errors='coerce')
            dat = dat.set_index(time)
        dat = dat[var_id]
        return dat

    scen = scenarios.copy()
    if "historical" in scenarios:
        rcp_dfs_historical = {}
        scen.remove("historical")
        for models in model_list:
            data_all = []
            columns = []
            data_conc = []
            for v, v_id in zip(var_name, var_id):
                data_list = []
                for file in sorted(glob.glob(path + '/' + v + '/' + 'historical' + '/' + models + '/' + '*.nc')):
                #for file in sorted(glob.glob(path + '/' + v + '/' + 'historical' + '/' + '*.nc')): # testing
                    data_list.append(cmip_open(file, lat=lat, lon=lon, var_id=v_id, method=method))
                data_conc.append(pd.concat(data_list))
                data_name = [str(v_id) + '_' + 'historical']
                columns = columns + data_name
            data_all.append(pd.concat(data_conc, axis=1))
            rcp_data = pd.concat(data_all, axis=1)
            rcp_data.columns = columns
            rcp_data = rcp_data.dropna()
            rcp_data = rcp_data.sort_index().loc[start_date:end_date]
            rcp_data["pr_historical"] = rcp_data["pr_historical"]*86400
            rcp_dfs_historical[models] = rcp_data

    for models in model_list:
        data_all = []
        columns = []
        for s in scen:
            data_conc = []
            for v, v_id in zip(var_name, var_id):
                data_list = []
                for file in sorted(glob.glob(path + '/' + v + '/' + s + '/' + models + '/' + '*.nc')):
                #for file in sorted(glob.glob(path + '/' + v + '/' + s + '/' + '*.nc')): #testing
                    data_list.append(cmip_open(file, lat=lat, lon=lon, var_id=v_id, method=method))
                data_conc.append(pd.concat(data_list))
                data_name = [str(v_id) + '_' + str(s)]
                columns = columns + data_name
            data_all.append(pd.concat(data_conc, axis=1))
            rcp_data = pd.concat(data_all, axis=1)
            rcp_data.columns = columns
            rcp_data = rcp_data.dropna()
            rcp_data = rcp_data.sort_index().loc[start_date:end_date]
            for y in rcp_data.loc[:, rcp_data.columns.str.startswith('pr')].columns:
                rcp_data[y] = rcp_data[y] * 86400 # prec in kg m-2 s-2
        if "historical" in scenarios:
            historical_df = rcp_dfs_historical[models].copy()
            for l in range(len(scen)-1):
                historical_df["tas_"+str(l)] = historical_df["tas_historical"]
                historical_df["pr_" + str(l)] = historical_df["pr_historical"]
            columns = rcp_data.columns
            historical_df.columns = columns
            rcp_data = historical_df.append(rcp_data)

        rcp_dfs[models] = rcp_data
    return rcp_dfs

rcp_dfs = cmip_csv(wd, scen, model_list, variable, var, lat, lon, start_date, end_date)

if write_files:
    ## save dataframes
    for i in rcp_dfs.keys():
        rcp_dfs[i].to_csv(output + i + "_" + str(lat) + "-" + str(lon) + "_" + str(start_date[:4]) + "-" + str(end_date[:4]) + ".csv")

## calculate mean
if "historical" in scen:
    scen.remove("historical")

mean_dict = {}
for i in scen:
    mean_dict[i] = pd.DataFrame()
    for v in variable:
        for k in rcp_dfs.keys():
            mean_dict[i][str(v) + "_" + str(k)] = rcp_dfs[k][str(v) + "_" + str(i)]
            mean_dict[i][str(v) + "_" + str(k)] = rcp_dfs[k][str(v) + "_" + str(i)]
        mean_dict[i][str(v) + "_" + "mean"] = mean_dict[i].filter(regex=str(v)).mean(axis=1)

if write_files:
    for i in mean_dict.keys():
        mean_dict[i].to_csv(output + str(i) + "_" + str(lat) + "-" + str(lon) + "_" + start_date + "-" + end_date + ".csv")



mean_df = pd.DataFrame(index=mean_dict[i].index)
for i in mean_dict.keys():
    if i.endswith("2_6"):
        mean_df["temp_26"] = mean_dict[i]["tas_mean"]
        mean_df["prec_26"] = mean_dict[i]["pr_mean"]
    if i.endswith("4_5"):
        mean_df["temp_45"] = mean_dict[i]["tas_mean"]
        mean_df["prec_45"] = mean_dict[i]["pr_mean"]
    if i.endswith("7_0"):
        mean_df["temp_70"] = mean_dict[i]["tas_mean"]
        mean_df["prec_70"] = mean_dict[i]["pr_mean"]
    if i.endswith("8_5"):
        mean_df["temp_85"] = mean_dict[i]["tas_mean"]
        mean_df["prec_85"] = mean_dict[i]["pr_mean"]

if write_files:
    mean_df.to_csv(output + "CMIP6_mean_" + str(lat) + "-" + str(lon) + "_" + start_date + "-" + end_date + ".csv")

## multiple plots
if plot:
    for i in scen:
        mean_dict[i] = mean_dict[i].resample("Y").agg({"tas_mean": "mean", "pr_mean": "sum"})
        for v in variable:
            for key in rcp_dfs.keys():
                if v == "tas":
                    yearly_df = rcp_dfs[key].resample("Y").agg("mean")
                if v == "pr":
                    yearly_df = rcp_dfs[key].resample("Y").agg("sum")
                plt.plot(yearly_df.index.to_pydatetime(), yearly_df[str(v) + "_" +str(i)], label=key, alpha=0.8)
                if v == "tas":
                    plt.title("CMIP6 yearly mean temperature for the " + str(i) + " scenario", size=10)
                if v == "pr":
                    plt.title("CMIP6 yearly precipitation sum for the " + str(i) + " scenario", size=10)
                plt.xlabel("Date")
                if v == "tas":
                    plt.ylabel("Temperature [K]")
                if v == "pr":
                    plt.ylabel("Precipitation [mm]")
            plt.plot(yearly_df.index.to_pydatetime(), mean_dict[i][str(v) + "_" + "mean"], label="mean", color="k")
            plt.legend()

            plt.show()

## all in one plot
if plot:
    if "historical" in scen:
        scen.remove("historical")

    for v in variable:
        fig, axs = plt.subplots(len(scen), sharex=True, sharey=True)
        if v == "tas":
            fig.suptitle("CMIP6 yearly mean temperature")
        if v == "pr":
            fig.suptitle("CMIP6 yearly precipitation sum")
        for s, i in zip(scen, range(len(scen))):
            mean_dict[s] = mean_dict[s].resample("Y").agg({"tas_mean": "mean", "pr_mean": "sum"})
            for key in rcp_dfs.keys():
                if v == "tas":
                    yearly_df = rcp_dfs[key].resample("Y").agg("mean")
                if v == "pr":
                    yearly_df = rcp_dfs[key].resample("Y").agg("sum")
                axs[i].plot(yearly_df.index.to_pydatetime(), yearly_df[v + "_" + str(s)], label=key, alpha=0.8)
                axs[i].set_title(s + " scenario", fontsize=9)
                if v == "tas":
                    axs[i].set_ylabel("Temperature [K]")
                if v == "pr":
                    axs[i].set_ylabel("Precipitation [mm]")
            axs[i].plot(yearly_df.index.to_pydatetime(), mean_dict[s][v + "_" + "mean"], label="mean", color="k")

        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                          fancybox=True, shadow=True, ncol=7)
        plt.show()
