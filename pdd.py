#!/usr/bin/env python

import numpy as np
from netCDF4 import Dataset as NC

# PDD model computations

def pdd(temp):
		"""Compute positive degree days from temperature time series"""
		return sum(np.greater(temp,0)*temp)*365.242198781/12

# netCDF IO

def init():
		"""Create an artificial PISM atmosphere file"""

		from math import cos, pi

		# open netcdf file
		nc = NC('atm.nc', 'w')

		# create dimensions
		tdim = nc.createDimension('time', 12)
		xdim = nc.createDimension('x', 8)
		ydim = nc.createDimension('y', 8)

		# prepare coordinate arrays
		x = range(len(xdim))
		y = range(len(ydim))
		(xx, yy) = np.meshgrid(x, y)

		# create air temperature variable
		temp = nc.createVariable('air_temp', 'f4', ('time', 'x', 'y'))
		temp.units = 'degC'

		# create precipitation variable
		prec = nc.createVariable('precipitation', 'f4', ('time', 'x', 'y'))
		prec.units = "m yr-1"

		# assign temperature and precipitation values
		for i in t:
			temp[i] = xx + 10 * cos(i*2*pi/12)
			prec[i] =      yy * cos(i*2*pi/12)

		# close netcdf file
		nc.close()

def main():
		"""Read atmosphere file and output surface mass balance"""

		# open netcdf files
		i = NC('atm.nc', 'r')
		o = NC('clim.nc', 'w')

		# read input data
		temp = i.variables['air_temp']
		prec = i.variables['precipitation']

		# create dimensions
		tdim = o.createDimension('time', 12)
		xdim = o.createDimension('x', 8)
		ydim = o.createDimension('y', 8)

		# compute the number of positive degree days
		pddvar = o.createVariable('pdd', 'f4', ('x', 'y'))
		print pddvar[:].shape
		pddvar[:] = pdd(temp)

# Called at execution

if __name__ == "__main__":

		# prepare dummy input dataset
		init()

		# run the mass balance model
		main()

