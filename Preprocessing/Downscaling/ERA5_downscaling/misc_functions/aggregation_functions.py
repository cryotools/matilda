def var_mean_time(var,step):
    var_mean=var.resample(time=step, keep_attrs = True).mean(dim='time', keep_attrs = True)
    return var_mean

def var_sum_time(var,step):
    var_mean=var.resample(time=step, keep_attrs = True).sum(dim='time', keep_attrs = True)
    return var_mean

def var_mean_time_total(var):
    var = var.mean(dim='time', keep_attrs=True)
    return var

def var_sum_time_total(var):
    var = var.sum(dim='time', keep_attrs=True)
    return var

def var_mean_space(var):
    #var = var.mean(dim=['south_north','west_east'],keep_attrs=True)
    var = var.mean(dim=['lat', 'lon'],keep_attrs=True)
    return var

def dataset_space_mean_save(dataset,path):
    #dataset_mean=dataset.mean(dim=['south_north','west_east'],keep_attrs=True)
    dataset_mean=dataset.mean(dim=['lat','lon'],keep_attrs=True)
    dataset_mean.to_netcdf(path)

def dataset_time_mean_save(dataset,step,path,keep_attrs=True):
    mean_dataset = dataset.resample(time=step).mean()
    mean_dataset.to_netcdf(path)

def dataset_time_sum_save(dataset,step,path):
    mean_dataset = dataset.resample(time=step,keep_attrs=True).sum()
    mean_dataset.to_netcdf(path)


