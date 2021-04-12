## Packages
from pathlib import Path; home = str(Path.home())
import numpy as np
import matplotlib.pyplot as plt
from pysheds.grid import Grid
import fiona
import geopandas as gpd
import gdal
import warnings
warnings.filterwarnings('ignore')

##
DEM_file = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/ALPSMLC30_N042E078_DSM.tif'
output_file = home + "/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/"

# Specify discharge point
x, y = 78.06284861, 42.32793047 # hydro
x2, y2 =  78.26573705, 42.16429283 # hydro glacier

##
grid = Grid.from_raster(DEM_file, data_name='dem')

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
#demView = grid.view('dem', nodata=np.nan)
#plotFigure(demView,'Elevation')
#plt.show()


## Create shapefile and save it
shapes = grid.polygonize()

schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

with fiona.open(output_file + "shapefile_hydro.shp", 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1
c.close()

##
grid = Grid.from_raster(DEM_file, data_name='dem')

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
grid.catchment(data='dir', x=x2, y=y2, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)
# Clip the bounding box to the catchment
grid.clip_to('catch')
# Get a view of the catchment
#demView = grid.view('dem', nodata=np.nan)
#plotFigure(demView,'Elevation')
#plt.show()


## Create shapefile and save it
shapes = grid.polygonize()

schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

with fiona.open(output_file + "shapefile_hydro_glacier.shp", 'w',
                driver='ESRI Shapefile',
                crs=grid.crs.srs,
                schema=schema) as c:
    i = 0
    for shape, value in shapes:
        rec = {}
        rec['geometry'] = shape
        rec['properties'] = {'LABEL' : str(value)}
        rec['id'] = str(i)
        c.write(rec)
        i += 1
c.close()