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
DEM_file = home + '/Seafile/Ana-Lena_Phillip/data/input_output/input/observations/karabatkak/static/srtm_catchment_clip.tif'
output_file = home + "/Seafile/Papers/No1_Kysylsuu_Bash-Kaingdy/data/GIS/Kysylsuu/Catchment_shapefile_new.shp"

# Specify discharge point
y, x = 42.32703430, 78.06141496 # hydro klappt (selbst gewählt)
y,x = 42.30029106, 78.09146228 # hydro new

##
grid = Grid.from_raster(DEM_file, data_name='dem')

#Define a function to plot the digital elevation model
def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12,10))
    plt.imshow(data, extent=grid.extent)
    plt.colorbar(label=label)
    plt.grid()
#Minnor slicing on borders to enhance colobars
elevDem = grid.dem[:-1,:-1]
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
grid.catchment(data=grid.dir, x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)
# Clip the bounding box to the catchment
grid.clip_to('catch')
# Get a view of the catchment
demView = grid.view('dem', nodata=np.nan)
plotFigure(demView, 'Elevation')
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
