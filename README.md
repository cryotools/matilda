# MATILDA - Modeling Water Resources in Glacierized Catchments

The MATILDA framework combines a simple positive degree-day routine (DDM) to compute glacial melt with the hydrological bucket model HBV (Bergström, 1986). The aim is to provide an easy-access open-source tool to assess the characteristics of small and medium-sized glacierized catchments and enable users to estimate future water resources for different climate change scenarios.
MATILDA is an ongoing project and therefore a work in progress.

## Overview

In the basic setup, MATILDA uses a modified version of the [pypdd](https://github.com/juseg/pypdd.git) tool to calculate glacial melt based on a positive degree-day  approach and a modified version of HBV from the Lumped Hydrological Models Playground ([LHMP](https://github.com/hydrogo/LHMP.git)). The output contains the modeled time series for various components of the water balance, basic statistics for these variables, a choice of two model effieciency coefficients (NSE, KGE), and several plots of in- and output data.

![](/MATILDA/MATILDA_slim/workflow_detailed-CORRECTED.png)

### Requirements

The tool should run with every Python3 version on all computer operating systems. It was developed on Python 3.6.9 on Ubuntu 18.04.
It requires the following Python3 libraries:
- xarray
- numpy
- pandas
- matplotlib
- scipy
- os
- datetime
- hydroeval

The MATILDA package and the necessary packages can be installed to your local machine by using pip or a comparable package manager. You can either install the package by using the link to this repository:
```
pip install git+https://git@github.com/cryotools/matilda.git

```
Or clone this repository to you local machine, navigate to the top directory, and use:
```
pip install .
```

### Data

The minimum input is a CSV-file containing timeseries of air temperature (°C), total precipitation (mm) and (if available) evapotranspiration (mm) data in the  format shown below. If Evapotranspiration is not provided it is caclulated from air temperature following [Oudin et.al. 2010](https://doi.org/10.1080/02626660903546118). A series of runoff observations (mm) is used to calibrate/validate the model. All data sets need at least daily resolution.

| TIMESTAMP            | T2            | RRR            | PE            |
| -------------        | ------------- | -------------  | ------------- |
| 2011-01-01 00:00:00  | -18.2         | 0.00           | 0.00          |
| 2011-01-01 01:00:00  | -18.3         | 0.1            | 0.00          |
| 2011-01-01 02:00:00  | -18.2         | 0.1            | 0.00          |

| Date          | Qobs          |
| ------------- | ------------- |
| 2011-01-01    | 0.17          |
| 2011-01-01    | 0.19          |


The forcing data is scaled to the mean glacier elevation and the mean catchment elevation respectively using linear lapse rates. Reference altitudes for the input data, the whole catchment, and the glacierized fraction need to be provided. Automated routines for catchment delineation and the download of public glacier data will be added to MATILDA in future versions.

To include the deltaH parameterization from [Huss and Hock 2010] () within the DDM routine to calculate glacier area evolution over the study period, information on the glaciers is necessary. The routine needs a initial glacier profile where the glaciers are divided into individual elevation bands which describes the glaciers at the beginning of the study period in form of a dataframe:
| Elevation     | Area         | WE            | EleZone       |
| ------------- | ------------ | ------------- | ------------- |
| 3720 		| 0.005        | 10786.061     | 3700	       |
| 3730  	| 0.001        | 13687.801     | 3700 	       |
| 3740  	| 0.001        | 12571.253     | 3700 	       |
| 3750  	| 0.002        | 12357.987     | 3800 	       |

Elevation shows the elevation of each elevation bands (10 m zones are recommended), Area the area of each band as a fraction of the whole glacier area, WE the ice thickness in m w.e. and EleZone the combinded bands over 100-200 m.

### Workflow

The MATILDA package consists of four different modules: parameter setup, data preprocessing, core simulation, and postprocessing. All modules can be used individually or via the superior *MATILDA_simulation* function. 
To use the whole package, the following steps are recommended:
- Load your data.
- Define the spin-up and simulation periods. At least one year of spin-up is recommended.
- Specify your catchment properties (catchment area, glacierized area, average elevation, average glacier elevation).
- Define the output frequency (daily, weekly, monthly or yearly).
- Specify parameters as you please using the *MATILDA_parameter* function. If no parameters are specified, default values are applied.
- Run the data preprocessing with the *MATILDA_preproc* function.
- Run the actual simulation with the *MATILDA_submodules* function.
- The simulation will give you a quick overview of your output and (if you provide observations), model efficiency coefficients are calculated.
- Plot runoff, meteorological parameters, and HBV output variables using *MATILDA_plots* function. 
- All the output including the plots and parameters can be saved to your local disk with the *MATILDA_save_output* function.

An example script for the workflow can be found [here](MATILDA/example_workflow.py).

## Built using
* [pypdd](https://github.com/juseg/pypdd.git) - Python positive degree day model for glacier surface mass balance
* [LHMP](https://rometools.github.io/rome/) - Lumped Hydrological Models Playgroud - HBV Model

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
	•	Ayzel, G. (2016). Lumped Hydrological Models Playground. [github.com/hydrogo/LHMP](https://github.com/hydrogo/LHMP.git), [doi:10.5281/zenodo.59680](https://doi.org/10.5281/zenodo.59680).
	•	Ayzel G. (2016). LHMP: lumped hydrological modelling playground. Zenodo. [doi:10.5281/zenodo.59501](https://doi.org/10.5281/zenodo.59501).
	•	Bergström, S. (1992). The HBV model: Its structure and applications. Swedish Meteorological and Hydrological Institute.
