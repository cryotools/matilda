## Packages
from pathlib import Path; home = str(Path.home())
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pysheds.grid import Grid
import fiona
import geopandas as gpd
import subprocess
import warnings
warnings.filterwarnings('ignore')

## Input
working_directory = home + "/Seafile/SHK/Scripts/centralasiawaterresources/Catchment/"
input_DEM = home + "/Seafile/Ana-Lena_Phillip/data/input_output/static/DEM/n43_e086_3arc_v2.tif"
RGI_files = home + "/Seafile/Tianshan_data/GLIMS/13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp"
ice_thickness_files = home + "/Seafile/Tianshan_data/01_original/"

x, y = 86.82151540, 43.11473921 # pouring point
ele_bands, ele_zones = 20, 100

output_path = home + "/Seafile/Ana-Lena_Phillip/data/input_output/static/GIS_routine/"

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
plt.show()

# Detect depressions
depressions = grid.detect_depressions('dem')

# Plot depressions
plt.imshow(depressions)
plt.show()


# Fill depressions
grid.fill_depressions(data='dem', out_name='flooded_dem')
# Test result
depressions = grid.detect_depressions('flooded_dem')
plt.imshow(depressions)
plt.show()

# Detect flats
flats = grid.detect_flats('flooded_dem')

# Plot flats
plt.imshow(flats)
plt.show()

grid.resolve_flats(data='flooded_dem', out_name='inflated_dem')
plt.imshow(grid.inflated_dem[:-1,:-1])
plt.show()


# Create a flow direction grid
#N    NE    E    SE    S    SW    W    NW
dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)
plotFigure(grid.dir,'Flow Direction','viridis')
plt.show()


# Delineate the catchment
grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)
# Clip the bounding box to the catchment
grid.clip_to('catch')
# Get a view of the catchment
demView = grid.view('dem', nodata=np.nan)
plotFigure(demView, 'Elevation')
plt.show()

# save as clipped TIF
grid.to_raster(demView, output_path + "catchment_DEM.tif")
print("Mean catchment elevation is " + str(np.nanmean(demView)) + " m")

# Create shapefile and save it
shapes = grid.polygonize()
#
# schema = {
#     'geometry': 'Polygon',
#     'properties': {'LABEL': 'float:16'}
# }
#
# with fiona.open(output_path + "catchment_shapefile.shp", 'w',
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
catchment_shapefile = gpd.GeoDataFrame.from_file(output_path + "catchment_shapefile.shp")

glaciers_catchment = gpd.overlay(rgi_shapefile, catchment_shapefile, how='intersection')
fig, ax = plt.subplots(1, 1)
base = catchment_shapefile.plot(color='white', edgecolor='black')
glaciers_catchment.plot(ax=base, column="RGIId", legend=True)
plt.show()


## Copy/get ice thickness files for each glacier
glaciers_catchment.to_file(driver = 'ESRI Shapefile', filename= output_path + "result.shp")

glaciers_catchment.drop('geometry', axis=1).to_csv(working_directory + 'rgi_csv.csv', index=False) # gets deleted after

# here code for the shell script
os.chdir(path=working_directory)
subprocess.call(['./ice_thickness', ice_thickness_files, output_path])


## create shapefile with different elevation bands/lines and cut this to the glacier shapefiles
subprocess.call(["./elevation_bands", output_path + "catchment_DEM.tif", output_path + "catchment_contours.shp", str(ele_bands)])

contours_shapefile = gpd.GeoDataFrame.from_file(output_path + "catchment_contours.shp")
contours = gpd.clip(contours_shapefile, glaciers_catchment)
contours.to_file(output_path + "catchment_contours.shp")

## apply contour lines on ice thickness files and calculate mass per band
thickness_list = glob.glob(os.path.join(output_path, "*thickness.tif"))
with open(working_directory + 'thickness.txt', 'w+') as f:
    # write elements of list
    for items in thickness_list:
        f.write('%s\n' % items)
# close the file
f.close()

subprocess.call(["./ice_thickness_files", output_path])


##
file = "/home/ana/Desktop/Thickness/contours.shp"
output = "/home/ana/Desktop/Thickness/contours2.shp"
gdf1 = gpd.read_file(file)
gdf2 = gpd.read_file(output_shapefile)


gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2]))
gdf.plot()
plt.show()
## combine into elevation zones and create dataframe.