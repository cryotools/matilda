import sys
import os
import xarray as xr
from shapely.geometry import Point, mapping
import matplotlib.pyplot as plt
import rioxarray as rxr
import geopandas as gpd
import numpy as np
import salem
from pathlib import Path; home = str(Path.home())

lat = 41; lon = 75.9
start_date = '2000-01-01'; end_date = '2020-12-31'

##
shdf = salem.read_shapefile(home+ "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")
ds = salem.open_wrf_dataset(home +"/Desktop/HAR/HARv2_d10km_d_2d_prcp_2000.nc")

crop_extent = shdf.to_crs(ds.pyproj_srs)
crop_extent.crs = ds.pyproj_srs

new_shdf = crop_extent.copy()
#add buffer to geom, else it will not be detected within bounds of 10km spacing
new_shdf.geometry = new_shdf.geometry.buffer(6000)

clipped_ds = ds.salem.subset(shape=new_shdf)

ds_static = salem.open_wrf_dataset('/home/ana/Desktop/HAR/HAR v2_d10km_static_hgt.nc')
clipped_static = ds_static.salem.subset(shape=new_shdf)

fig = plt.figure(figsize=(16,12), dpi=300)
ax = fig.add_subplot(111)
shdf.plot(ax=ax, zorder=3)
new_shdf.plot(alpha=0.2, zorder=2, ax=ax)
clipped_ds.prcp.mean(dim='time').plot(ax=ax, zorder=-1)
plt.show()

### Convert shapefile to wrf projection ###
wrf_ds = salem.open_wrf_dataset(home +"/Desktop/HAR/HARv2_d10km_d_2d_prcp_2000.nc")
crop_extent_lat_lon = gpd.read_file(home+ "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/shapefile_hydro_kyzylsuu.shp")
crop_extent_lat_lon = crop_extent_lat_lon.to_crs(wrf_ds.pyproj_srs)
crop_extent_lat_lon.crs = wrf_ds.pyproj_srs
crop_extent_lat_lon.to_file(har_path+'abramov_har_proj.shp')



### The same thing can be done using salem ###
salem_xds = salem.open_xr_dataset(home +"/Desktop/HAR/HARv2_d10km_d_2d_prcp_2000.nc")

new_test = salem_xds.salem.subset(shape=new_shdf)
ax=new_shdf.plot(alpha=0.2)
shdf.plot(ax=ax, zorder=6)
new_test.prcp.mean(dim='time').plot(ax=ax, zorder=-1)

new_test_v2 = new_test.where(new_test.south_north > 900000, drop=True)
new_test_v2
ax=new_shdf.plot(alpha=0.2)
shdf.plot(ax=ax, zorder=6)
new_test_v2.prcp.mean(dim='time').plot(ax=ax, zorder=-1)
