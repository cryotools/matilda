# -*- coding: utf-8 -*-
'''
Copyright (c) 2024 by Phillip Schuster
This file is part of the framework for Modeling Water Resources in Glacierized Catchments (MATILDA).

:author: Phillip Schuster, Ana-Lena Tappe & Alexander Georgi

This package combines a positive degree-day model for glacier and snow melt with an adapted version of
the lumped conceptual catchment model HBV. It further features a glacier area-volume scaling routine to account for
glacier changes.

:dependencies:
    - HydroErr==1.24
    - hydroeval==0.1.0
    - matplotlib==3.9.2
    - numpy==1.26.4
    - pandas==2.2.3
    - plotly==5.24.1
    - scipy==1.10.1
    - xarray==2024.10.0
    - DateTime==5.5
    - pyyaml==6.0.2
    - spotpy==1.6.2
    - SciencePlots==2.1.1
'''
from . import core, mspot_glacier  # Contains all core functions

__version__ = '1.0'

