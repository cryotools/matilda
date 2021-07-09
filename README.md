# Hydrological modeling routine to assess water ressources in glacierized catchments (under conditions of climate change)
This tool connects the output of the glacier mass balance model COSIPY (COupled Snow and Ice energy and MAss Balance in Python) with the HBV model (Bergström, 1986), a simple hydrological bucket model, which computes runoff and a simple DDM approach to compute the glacier melt. The aim is to generate runoff projections under different climate scenarios and use the results to help planing future water management strategies in the modeled catchments. 

## Overview

The tool uses the output of the COSIPY model, the translation of the COSIMA model into python (https://github.com/cryotools/cosipy.git), a modified version of the pypdd tool (https://github.com/juseg/pypdd.git) to calculate runoff from the glacier(s) with a simple DegreeDayModel approach and a modified version of the LHMP tool (https://github.com/hydrogo/LHMP.git) which translates the HBV model into python. 

### Requirements

Clone
```
Clone this repo to your local machine using https://scm.cms.hu-berlin.de/sneidecy/centralasiawaterresources.git
```

The tool should run with any Python3 version on any computer operating system. It was developed on Python 3.6.9 on Ubuntu 18.04.
It requires the following Python3 libraries:
- 	xarray
- 	numpy
- 	pandas
- 	matplotlib  


### Data

The necessary input for the tool are the two input files created by the preprocessing script of the Cosipy model, a netcdf and a csv file which consists of a time series of various meteorological variables and a data frame with observational runoff data (csv). 
It is also possible to use input apart from the COSIPY model. To run the tool, a netcdf and a data frame with a timeseries of temperature, precipitation and a mask of the glacier areas are sufficient (for the netcdf). 

It is also necessary to adjust the parameters of the DDM and the HBV model to the prevailing conditions in the test area. 

### Workflow

To run the tool, please make your adjustments in the file ConfigFile.py. You can then use the bash script … to run the tool.
The tool consists of two main scripts to generate the projected runoff and build plots for a first overview of the data. 

The model script first uses the netcdf dataset to calculate glacier runoff with the help of a simple DDM approach. For this step, the glacier mask is needed to ensure inclusion of only glaciered area. The DegreeDayModel then uses the average temperature of each day to calculate the positive degree days(PDD), the temperature sum over the given threshold. It then computes the snow fraction in the given time period and uses the PDD and snow variables to compute snow and ice melt on the glacier area. 

To calculate the runoff from the overall catchment area, the data frame with temperature, precipitation and evapotranspiration is needed to run the HBV implementation. If evapotranspiration data is not available, it will be calculated with a formula by Oudin et. et al. (2005). The HBV model consists of different subroutines to estimate snow accumulation and melt, evapotranspiration, soil moisture and the runoff generation. Various parameters need to be adjusted to the test area in the ConfigFile. 

The output of the model is a csv data frame which consists of the need meteorological parameters (temperature, precipitation and evapotranspiration) and the calculated runoff from the DDM and the HBV as well as the final total runoff. Plots to compare input parameters as well as the runoff are also generated. 

## Built With
* [Python](https://www.python.org) - Python
* [COSIPY](https://github.com/cryotools/cosipy.git) - COupled Snow and Ice energy and MAss Balance in Python
* [pypdd](ttps://github.com/juseg/pypdd.git) - Python positive degree day model for glacier surface mass balance
* [LHMB](https://rometools.github.io/rome/) - Lumped Hydrological Models Playgroud - HBV Model

## Authors

* **Phillip Schuster** - *Initial work* - (https://scm.cms.hu-berlin.de/schustep)
* **Ana-Lena Tappe** - *Initial work* - (https://scm.cms.hu-berlin.de/tappelen)


See also the list of [contributors](https://scm.cms.hu-berlin.de/sneidecy/centralasiawaterresources/-/graphs/master) who participated in this project.

## License

This project is licensed under the HU Berlin License - see the [LICENSE.md](LICENSE.md) file for details

### References

For COSIPY:
	•	Sauter, T. & Arndt, A (2020). COSIPY – An open-source coupled snowpack and ice surface energy and mass balance model. https://doi.org/10.5194/gmd-2020-21

For PyPDD:
	•	Seguinot, J. (2019). PyPDD: a positive degree day model for glacier surface mass balance (Version v0.3.1). Zenodo. http://doi.org/10.5281/zenodo.3467639

For LHMP and HBV:
	•	Ayzel, G. (2016). Lumped Hydrological Models Playground. github.com/hydrogo/LHMP, hub.docker.com/r/hydrogo/lhmp/, doi: 10.5281/zenodo.59680.
	•	Ayzel G. (2016). LHMP: lumped hydrological modelling playground. Zenodo. doi: 10.5281/zenodo.59501.
	•	Bergström, S. (1992). The HBV model: Its structure and applications. Swedish Meteorological and Hydrological Institute.
