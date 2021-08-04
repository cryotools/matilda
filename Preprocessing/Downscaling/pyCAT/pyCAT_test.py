from pycat.io import Dataset
from pycat.esd import QuantileMapping
from pycat.esd import ScaledDistributionMapping
import iris
from iris.pandas import as_cube, as_series, as_data_frame
from cf_units import Unit


import warnings
warnings.filterwarnings("ignore")  # sklearn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.stats as stats
from pathlib import Path
import sys
import socket
host = socket.gethostname()
if 'node' in host:
    home = '/data/projects/ebaca'
elif 'cirrus' in host:
    home = '/data/projects/ebaca'
else:
    home = str(Path.home()) + '/Seafile'
wd = home + '/Ana-Lena_Phillip/data/matilda/Preprocessing'
import os
os.chdir(wd + '/Downscaling')
sys.path.append(wd)
from Preprocessing_functions import pce_correct, trendline, daily_annual_T
from forcing_data_preprocessing_kyzylsuu import era_temp_D_int, aws_temp_D_int, era_temp_D, aws_temp_D, era,\
    aws_temp, cmip, aws_prec, era_D

aws_temp_D.to_csv('pyCAT/sample-data/aws.csv')

def boiler_plate(df, name=None, calendars=None,
                 unit=None, index_name=None):
    cube = as_cube(df, calendars=calendars)
    cube.rename(name)
    cube.coord("index").rename("time")
    cube.units = unit
    return cube

cube.

kw = dict(index_name='time',
          unit=Unit('C'))
          # , calendars={0: iris.unit.CALENDAR_GREGORIAN})

obs = boiler_plate(aws_temp_D, name='obs', **kw)
mod = boiler_plate(era_D['2007-08-10':'2016-01-01'], name='mod', **kw)
sce = boiler_plate(era_D['2000-01-01':'2020-12-31'], name= 'sce', **kw)

obs = Dataset('pyCAT/sample-data', 'observation.nc')
mod = Dataset('pyCAT/sample-data', 'model*.nc')
sce = Dataset('pyCAT/sample-data', 'scenario*.nc')
sdm = ScaledDistributionMapping(obs, mod, sce)
sdm.correct()

sdm = ScaledDistributionMapping(obs, mod, sce)
sdm.correct()