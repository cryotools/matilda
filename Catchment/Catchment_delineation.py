## Packages
from pathlib import Path; home = str(Path.home())
import numpy as np
import matplotlib.pyplot as plt
from pysheds.grid import Grid
import fiona
# import gdal
# import GDAL
import warnings
warnings.filterwarnings('ignore')

## Specify information about the data here
# Example Kashkator
DEM_file = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/srtm_catchment_clip.tif'
output_file = home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/GIS/Kysylsuu/kashkator_catchment.shp"

# Specify discharge point
y, x = 42.30029106, 78.09146228 # hydro new
y, x = 42.16268596, 78.26632477 # hydro Kashkator

##
#Define a function to plot the digital elevation model
def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12,10))
    plt.imshow(data, extent=grid.extent)
    plt.colorbar(label=label)
    plt.grid()

## Catchment delineation with pysheds
# https://github.com/mdbartos/pysheds

# Plot the DEM
grid = Grid.from_raster(DEM_file, data_name='dem')
grid.view('dem')

# Fill depressions in DEM
grid.fill_depressions('dem', out_name='flooded_dem')
# Resolve flats in DEM
grid.resolve_flats('flooded_dem', out_name='inflated_dem')

# Specify directional mapping
#N    NE    E    SE    S    SW    W    NW
dirmap = (64, 128, 1, 2, 4, 8, 16, 32)
# Compute flow directions
grid.flowdir(data='inflated_dem', out_name='dir', dirmap=dirmap)

# Delineate the catchment based on the pouring point
grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label')

# Clip the DEM to the catchment
grid.clip_to('catch')
demView = grid.view('dem', nodata=np.nan)
plotFigure(demView,'Elevation')
plt.show()

## Create shapefile and save it
shapes = grid.polygonize()

schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

with fiona.open(output_file, 'w',
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
print("Mean catchment elevation is " + str(np.nanmean(demView)) + " m")

## TEST Elevation bands with matplotlib
gdal_data = gdal.Open(DEM_file)
gdal_band = gdal_data.GetRasterBand(1)
nodataval = gdal_band.GetNoDataValue()

# convert to a numpy array
data_array = gdal_data.ReadAsArray().astype(np.float)

# replace missing values if necessary
if np.any(data_array == nodataval):
    data_array[data_array == nodataval] = np.nan

#Plot out data with Matplotlib's 'contour'
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(111)
plt.contour(data_array, cmap = "viridis",
            levels = list(range(0, 5000, 100)))
plt.title("Elevation Contours of Kyzylzuu catchment")
cbar = plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()
