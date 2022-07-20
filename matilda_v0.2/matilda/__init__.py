# -*- coding: utf-8 -*-
'''
Copyright (c) 2022 by Phillip Schuster
This file is part of the framework for Modeling Water Resources in Glacierized Catchments (MATILDA).

:author: Phillip Schuster & Ana-Lena Tappe

This package combines a positive degree-day model for glacier and snow melt with an adapted version of
the lumped conceptual catchment model HBV. It further features a glacier area-volume scaling routine to account for
glacier changes.

:dependencies: - Numpy >1.8 (http://www.numpy.org/)
               - Scipy >1.5 (https://pypi.org/project/scipy/)
               - Pandas >0.13 (optional) (http://pandas.pydata.org/)
               - Matplotlib >1.4 (optional) (http://matplotlib.org/)
               - CMF (optional) (http://fb09-pasig.umwelt.uni-giessen.de:8000/)
               - mpi4py (optional) (http://mpi4py.scipy.org/)
               - pathos (optional) (https://pypi.python.org/pypi/pathos/)
               - sqlite3 (optional) (https://pypi.python.org/pypi/sqlite3/)
               - numba (optional) (https://pypi.python.org/pypi/numba/)

               - Pandas
               - Xarray
               - Matplotlib
               - Numpy
               - Scipy
               - Datetime
               - Hydroeval

MATILDA is still in an experimental state and a work in progress.
'''
from . import matilda_core            # Writes the results of the sampler in a user defined output file

__version__ = '0.2'