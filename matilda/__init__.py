# -*- coding: utf-8 -*-
'''
Copyright (c) 2022 by Phillip Schuster
This file is part of the framework for Modeling Water Resources in Glacierized Catchments (MATILDA).

:author: Phillip Schuster & Ana-Lena Tappe

This package combines a positive degree-day model for glacier and snow melt with an adapted version of
the lumped conceptual catchment model HBV. It further features a glacier area-volume scaling routine to account for
glacier changes.

:dependencies: - Pandas
               - Xarray
               - Matplotlib
               - Numpy
               - Scipy
               - Datetime
               - Hydroeval

MATILDA is still in an experimental state and a work in progress.
'''
from . import core, mspot_glacier            # Contains all core functions

__version__ = '0.3'
