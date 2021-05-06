import xarray as xr
import pandas as pd
from pathlib import Path
import glob
import subprocess
import os
home = str(Path.home())
wd = home + "/Seafile/Tianshan_data/CMIP/CMIP6/dir_test"

## output information
var = ['near_surface_air_temperature', 'precipitation']
variable = ["tas", "pr"]
scen = ['ssp1_2_6', 'ssp2_4_5']
start_date = '2020-01-01'; end_date = '2029-12-31'
lat = 42.25; lon = 78.25

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


## find models: bash version
# os.putenv('EXPERIMENT', ' '.join(scen))
# os.putenv('MODEL', ' '.join(model))
#
# os.chdir(path="/home/ana/Desktop/")
# subprocess.call(['./cmip_test3', working_directory, str(lon), str(lat)])

##
def cmip_csv(path, scenarios, model_list, var_id, var_name, lat, lon, start_date, end_date, method='nearest'):
    rcp_dfs = {}

    def cmip_open(path, lat, lon, var_id, method):
        ds = xr.open_dataset(path)
        pick = ds.sel(lat=lat, lon=lon, method=method)
        dat = pick[[var_id]].to_dataframe()
        if not isinstance(dat.index, pd.DatetimeIndex):
            time = ds.indexes['time'].to_datetimeindex()
            dat = dat.set_index(time)
        dat = dat[var_id]
        return dat

    for models in model_list:
        data_all = []
        columns = []
        for s in scenarios:
            data_conc = []
            for v, v_id in zip(var_name, var_id):
                data_list = []
                for file in sorted(glob.glob(path + '/' + v + '/' + s + '/' + models + '/' + '*.nc')):
                    data_list.append(cmip_open(file, lat=lat, lon=lon, var_id=v_id, method=method))
                data_conc.append(pd.concat(data_list))
                data_name = [str(v_id) + '_' + str(s)]
                columns = columns + data_name
            data_all.append(pd.concat(data_conc, axis=1))
            rcp_data = pd.concat(data_all, axis=1)
            rcp_data.columns = columns
            rcp_data = rcp_data[start_date:end_date]
        rcp_dfs[models] = rcp_data
    return rcp_dfs

rcp_dfs = cmip_csv(wd, scen, model_list, variable, var, lat, lon, start_date, end_date)

# ## save dataframes
# for i in rcp_dfs.keys():
#     rcp_dfs[i].to_csv(wd + "/CMIP6_" + i + "_" + str(lat) + "-" + str(lon) + "_" + str(start_date[:4]) + "-" + str(end_date[:4]) + ".csv")

## multiple plots
import matplotlib.pyplot as plt
mean_dict = {}
for i in scen:
    mean_dict[i] = pd.DataFrame()
    for k in rcp_dfs.keys():
        mean_dict[i]["tas_" + str(k)] = rcp_dfs[k]["tas_" + str(i)]
    mean_dict[i]["mean"] = mean_dict[i].mean(axis=1)
    mean_dict[i] = mean_dict[i].resample("Y").agg("mean")
    for key in rcp_dfs.keys():
        yearly_df = rcp_dfs[key].resample("Y").agg("mean")
        plt.plot(yearly_df.index.to_pydatetime(), yearly_df["tas_"+str(i)], label=key)
        plt.title("CMIP6 yearly mean temperature for the " + str(i) + " scenario", size=10)
        plt.xlabel("Date")
        plt.ylabel("Temperature [K]")
    plt.plot(yearly_df.index.to_pydatetime(), mean_dict[i]["mean"], label="mean", color="k")
    plt.legend()

    plt.show()

## all in one plot
mean_dict = {}

fig, axs = plt.subplots(len(scen), sharex=True, sharey=True)
fig.suptitle("CMIP6 yearly mean temperature")
for s, i in zip(scen, range(len(scen))):
    mean_dict[s] = pd.DataFrame()
    for k in rcp_dfs.keys():
        mean_dict[s]["tas_" + str(k)] = rcp_dfs[k]["tas_" + str(s)]
    mean_dict[s]["mean"] = mean_dict[s].mean(axis=1)
    mean_dict[s] = mean_dict[s].resample("Y").agg("mean")

    for key in rcp_dfs.keys():
        yearly_df = rcp_dfs[key].resample("Y").agg("mean")
        axs[i].plot(yearly_df.index.to_pydatetime(), yearly_df["tas_"+str(s)], label=key)
        axs[i].set_title(s + " scenario", fontsize=9)
        axs[i].set_ylabel("Temperature [K]")
    axs[i].plot(yearly_df.index.to_pydatetime(), mean_dict[s]["mean"], label="mean", color="k")

plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=10)
plt.show()

##

