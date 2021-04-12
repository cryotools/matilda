## Packages
from pathlib import Path; home = str(Path.home())
import os
import numpy as np
import matplotlib.pyplot as plt
from pysheds.grid import Grid
import fiona
import geopandas as gpd
import gdal
import subprocess
import warnings
warnings.filterwarnings('ignore')

## Input
input_DEM = home + "/Seafile/Ana-Lena_Phillip/data/input_output/static/DEM/n43_e086_3arc_v2.tif"
RGI_files = "/home/ana/Seafile/Tianshan_data/GLIMS/13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp"
x, y = 86.82151540, 43.11473921 # pouring point
ele_bands, ele_zones = 0,0

output = home + "/Seafile/Ana-Lena_Phillip/data/input_output/static/"

## Catchment delineation
grid = Grid.from_raster(input_DEM, data_name='dem')

#Define a function to plot the digital elevation model
def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12,10))
    plt.imshow(data, extent=grid.extent)
    plt.colorbar(label=label)
    plt.grid()
#Minnor slicing on borders to enhance colobars
elevDem=grid.dem[:-1,:-1]
plotFigure(elevDem, 'Elevation (m)')

# Detect depressions
depressions = grid.detect_depressions('dem')

# Plot depressions
plt.imshow(depressions)

# Fill depressions
grid.fill_depressions(data='dem', out_name='flooded_dem')
# Test result
depressions = grid.detect_depressions('flooded_dem')
plt.imshow(depressions)
# Detect flats
flats = grid.detect_flats('flooded_dem')

# Plot flats
plt.imshow(flats)

grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
plt.imshow(grid.inflated_dem[:-1,:-1])

# Create a flow direction grid
#N    NE    E    SE    S    SW    W    NW
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
plotFigure(grid.dir,'Flow Direction','viridis')

# Delineate the catchment
grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)
# Clip the bounding box to the catchment
grid.clip_to('catch')
# Get a view of the catchment
demView = grid.view('dem', nodata=np.nan)
plotFigure(demView,'Elevation')
plt.show()

# Create shapefile and save it
shapes = grid.polygonize()
#
# schema = {
#     'geometry': 'Polygon',
#     'properties': {'LABEL': 'float:16'}
# }
#
# with fiona.open(output + "Shapefiles/catchment_Urumqi.shp", 'w',
#                 driver='ESRI Shapefile',
#                 crs=grid.crs.srs,
#                 schema=schema) as c:
#     i = 0
#     for shape, value in shapes:
#         rec = {}
#         rec['geometry'] = shape
#         rec['properties'] = {'LABEL' : str(value)}
#         rec['id'] = str(i)
#         c.write(rec)
#         i += 1
# c.close()

## Cutting all glaciers within the catchment from the RGI shapefile
rgi_shapefile = gpd.GeoDataFrame.from_file(RGI_files)
catchment_shapefile = gpd.GeoDataFrame.from_file(output + "Shapefiles/catchment_Urumqi.shp")

glaciers_catchment = gpd.overlay(rgi_shapefile, catchment_shapefile, how='intersection')
fig, ax = plt.subplots(1, 1)
base = catchment_shapefile.plot(color='white', edgecolor='black')
glaciers_catchment.plot(ax=base, column="RGIId", legend=True)
plt.show()


## Copy/get ice thickness files for each glacier

glaciers_catchment.drop('geometry', axis=1).to_csv('/home/ana/Desktop/csv_test.csv', index=False)

# here code for the shell script
os.system('./Catchment/ice_thickness.sh')

subprocess.call(['sh', './home/ana/Desktop/ice_thickness.sh'])

## create shapefile with different elevation bands/lines and cut this to the glacier shapefiles

## apply contour lines on ice thickness files and calculate mass per band

## combine into elevation zones and create dataframe