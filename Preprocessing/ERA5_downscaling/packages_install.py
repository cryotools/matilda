import os
from pip._internal import main as pipmain

def import_or_install(name):
    try:
        print(name + ' already installed!')
        __import__(name)
    except ImportError:
        print(name + ' not installed, installing .....')
        pipmain(['install', name])

packages = ['geopandas', 'descartes','geopandas', 'numpy', 'joblib', 'pyproj', 'scipy', 'salem', 'xarray', 'statsmodels']

for package in packages:
    import_or_install(package)

import_or_install('geopandas')


