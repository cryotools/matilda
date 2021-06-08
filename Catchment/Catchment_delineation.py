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
DEM_file = home + '/Seafile/Masterarbeit/Data/QGIS/Delineation/DEM_clipped.tif'
#DEM_file = home + '/Seafile/Ana-Lena_Phillip/data/input_output/static/DEM/n43_e086_3arc_v2.tif'
output_file = home + "/Seafile/Masterarbeit/Bash_Kaindy/Delineation/catchment_Urumqi.shp"

# Specify discharge point
x, y = 75.953079,41.125814 # Bash Kaindy
#x, y = 86.82151540, 43.11473921 # Urumqi

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
demView = grid.view('dem', nodata=np.nan)
plotFigure(demView,'Elevation')
plt.show()

print("Mean catchment elevation is " + str(np.nanmean(demView)) + " m")

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

## work with shapefile
shp = gpd.read_file(home + "/Seafile/Masterarbeit/Bash_Kaindy/Delineation/catchment.shp")
fig, ax = plt.subplots(figsize=(6,6))
shp.plot(ax=ax)
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title('Catchment polygon')
plt.show()
## elevation bands
dem_catchment = home + "/Seafile/Ana-Lena_Phillip/PycharmProjects/Ana/clippedElevations.tif"
gdal_data = gdal.Open(dem_catchment)
gdal_band = gdal_data.GetRasterBand(1)
nodataval = gdal_band.GetNoDataValue()

# convert to a numpy array
data_array = gdal_data.ReadAsArray().astype(np.float)
data_array

# replace missing values if necessary
if np.any(data_array == nodataval):
    data_array[data_array == nodataval] = np.nan

#Plot out data with Matplotlib's 'contour'
fig = plt.figure(figsize = (12, 8))
ax = fig.add_subplot(111)
plt.contour(data_array, cmap = "viridis",
            levels = list(range(0, 5000, 100)))
plt.title("Elevation Contours of Urumqi catchment")
cbar = plt.colorbar()
plt.gca().set_aspect('equal', adjustable='box')
plt.show()