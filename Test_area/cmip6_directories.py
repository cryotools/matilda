import os


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
    models = []
    if isinstance(var, list):
        if isinstance(scen, list):
            for v in var:
                for s in scen:
                    path = list_files(wd + '/' + v + '/' + s + '/')
                    for p in path:
                        mod = p.split(s)[1].split('/')
                        models.append(mod[len(mod) - 2])
            models = list(set([i for i in models if models.count(i) >= len(var)*len(scen)]))

        else:
            for v in var:
                path = list_files(wd + '/' + v + '/' + scen + '/')
                for p in path:
                    mod = p.split(scen)[1].split('/')
                    models.append(mod[len(mod) - 2])
            models = list(set([i for i in models if models.count(i) >= len(var)]))

    elif isinstance(scen, list):
        for s in scen:
            path = list_files(wd + '/' + var + '/' + s + '/')
            for p in path:
                mod = p.split(s)[1].split('/')
                models.append(mod[len(mod) - 2])
        models = list(set([i for i in models if models.count(i) >= len(scen)]))

    else:
        path = list_files(wd + '/' + var + '/' + scen + '/')
        models = []
        for p in path:
            mod = p.split(scen)[1].split('/')
            models.append(mod[len(mod) - 2])

    return models


wd = '/data/projects/ensembles/cmip6'
var = ['near_surface_air_temperature', 'precipitation']
scen = ['ssp1_2_6', 'ssp2_4_5']     # 'historical' 'ssp1_2_6' 'ssp2_4_5' 'ssp3_7_0' 'ssp5_8_5'

print(list_models(wd, var, scen))




