import os

wd = '/data/projects/ensembles/cmip6'
wd = '/home/phillip/Seafile/Tianshan_data/CMIP/CMIP6/dir_test'
var = ['near_surface_air_temperature', 'precipitation']
scen = ['ssp1_2_6', 'ssp2_4_5']     #  'historical', 'ssp1_2_6', 'ssp2_4_5', 'ssp3_7_0', 'ssp5_8_5'


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



