# MATILDA - Modeling Water Resources in Glacierized Catchments

The MATILDA framework combines a simple positive degree-day routine (DDM) for computing glacier melt with the simple hydrological bucket model HBV (Bergström, 1986). The aim is to provide an easy-access open-source tool to assess the characteristics of small and medium-sized glacierized catchments and enable useres to estimate their future water resources for different climate change scenarios.
MATILDA is an ongoing project and therefore a work in progress.

## Overview

In the basic setup, MATILDA uses a modified version of the pypdd tool (https://github.com/juseg/pypdd.git) to calculate runoff from the glacier(s) with a simple positive degree-day model approach and a modified version of the LHMP tool (https://github.com/hydrogo/LHMP.git). The comprehensive output contains the modeled time series for various components of the water balance, basic statistics of these variables, the Nash-Sutcliffe efficiency coefficient and optionally the KGE to evaluate the predictive skills of the model, and several plots of in- and output data.

![](/MATILDA/MATILDA_slim/workflow_detailed-Full.png)

### Requirements

Clone this repo to your local machine using https://github.com/cryotools/matilda.git


The tool should run with any Python3 version on any computer operating system. It was developed on Python 3.6.9 on Ubuntu 18.04.
It requires the following Python3 libraries:
- xarray
- numpy
- pandas
- matplotlib
- scipy
- os
- datetime
- hydroeval

The MATILDA package and the necessary packages can be installed to you local machine by using pip (or pip3). Just navigate into the cloned folder and use the following command
```
pip install .
```
or install the package directly from the source by using

```
pip install git+https://git@github.com/cryotools/matilda.git

```
### Data

The minimum input is a CSV-file containing timeseries of air temperature (°C), total precipitation (mm) and (if available) evapotranspiration (mm) data in the  format shown below. A series of runoff observations (mm) is used to validate the model output. At least daily data is required.

| TIMESTAMP            | T2            | RRR            | PE            |
| -------------        | ------------- | -------------  | ------------- |
| 2011-01-01 00:00:00  | -18.2         | 0.00           | 0.00          |
| 2011-01-01 01:00:00  | -18.3         | 0.1            | 0.00          |
| 2011-01-01 02:00:00  | -18.2         | 0.1            | 0.00          |

| Date          | Qobs          |
| ------------- | ------------- |
| 2011-01-01    | 0.00          |
| 2011-01-01    | 0.00          |


It is also necessary to adjust the parameters of the DDM and the HBV model to the prevailing conditions in the model area. Since the DDM model calculates the glacier melt, it is necessary to scale the input data to the glacier. In the most simple manner, this can be achieved by using a lapse rate for temperature and precipitation and the elevation difference between the reference altitudes of the data and the glacier.

### Workflow

The MATILDA package consists of four different modules: setting up the parameters, data preprocessing, the actual simulation and the plots. All modules can be used individually or as one routine called *MATILDA_simulation*. 
To use the whole package, the following steps are recommended:
- Read in your data and set the parameters with the parameter function *MATILDA_parameter*.
- Define the set up and simulation period. One year of setting up is recommended.
- Define properties like area and elevation for your catchment for the catchment and if part of the catchment glacier area (if not set it to 0). The elevation of your data is required for the downscaling.
- Define the output frequency (daily, weekly, monthly or yearly).
- Set all the parameters for the glacier and hydrological routines. If no parameters are set, the standart values are used.
- Run the data preprocessing with *MATILDA_preproc*.
- Run the actual simulation with *MATILDA_submodules*.
- The simulation will give you a quick overview over the data and if you have observations, the Nash–Sutcliffe model efficiency coefficient and KGE is calculated.
- Plot runoff, meteorological parameters, and HBV output series using the plots module *MATILDA_plots*. 
- All the output including the plots and parameters can be saved to your computer with the save_output function *MATILDA_save_output*.

An example script for the workflow can be found [here](MATILDA/example_workflow.py).

## Built using
* [Python](https://www.python.org) - Python
* [pypdd](https://github.com/juseg/pypdd.git) - Python positive degree day model for glacier surface mass balance
* [LHMB](https://rometools.github.io/rome/) - Lumped Hydrological Models Playgroud - HBV Model

## Authors

* **Phillip Schuster** - *Initial work* - (https://github.com/phiscu)
* **Ana-Lena Tappe** - *Initial work* - (https://github.com/anatappe)


See also the list of [contributors](https://scm.cms.hu-berlin.de/sneidecy/centralasiawaterresources/-/graphs/master) who participated in this project.

## License

This project is licensed under the HU Berlin License - see the [LICENSE.md](LICENSE.md) file for details

### References

For PyPDD:
	•	Seguinot, J. (2019). PyPDD: a positive degree day model for glacier surface mass balance (Version v0.3.1). Zenodo. http://doi.org/10.5281/zenodo.3467639

For LHMP and HBV:
	•	Ayzel, G. (2016). Lumped Hydrological Models Playground. github.com/hydrogo/LHMP, hub.docker.com/r/hydrogo/lhmp/, doi: 10.5281/zenodo.59680.
	•	Ayzel G. (2016). LHMP: lumped hydrological modelling playground. Zenodo. doi: 10.5281/zenodo.59501.
	•	Bergström, S. (1992). The HBV model: Its structure and applications. Swedish Meteorological and Hydrological Institute.
