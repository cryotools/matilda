from pathlib import Path; home = str(Path.home())
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import geopandas as gpd
from pysheds.grid import Grid
import mplleaflet
import fiona


import warnings
warnings.filterwarnings('ignore')

DEM_file = home + '/Seafile/Masterarbeit/Bash_Kaindy/Delineation/DEM_clipped.tif'
grid = Grid.from_raster(DEM_file, data_name='dem')

def plotFigure(data, label, cmap='Blues'):
    plt.figure(figsize=(12,10))
    plt.imshow(data, extent=grid.extent, cmap=cmap)
    plt.colorbar(label=label)
    plt.grid()
plotFigure(grid.dem, 'Elevation (m)')

dirmap = (64,  128,  1,   2,    4,   8,    16,  32)
grid.flowdir(data='dem', out_name='dir', dirmap=dirmap)
plotFigure(grid.dir,'Flow Directiom','viridis')

x, y = 75.953079, 41.125814

# Delineate the catchment
grid.catchment(data='dir', x=x, y=y, dirmap=dirmap, out_name='catch',
               recursionlimit=15000, xytype='label', nodata_out=0)
# Clip the bounding box to the catchment
grid.clip_to('catch')
# Get a view of the catchment
demView = grid.view('dem', nodata=np.nan)
plotFigure(demView,'Elevation')
plt.show()

shapes = grid.polygonize()

schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

with fiona.open(home + "/Seafile/Masterarbeit/Bash_Kaindy/Delineation/catchment.shp", 'w',
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

grid.accumulation(data='catch', dirmap=dirmap, pad_inplace=False, out_name='acc')
accView = grid.view('acc', nodata=np.nan)
plotFigure(accView,"Cell Number",'PuRd')
streams = grid.extract_river_network('catch', 'acc', threshold=200, dirmap=dirmap)
streams["features"][:2]
def saveDict(dic,file):
    f = open(file,'w')
    f.write(str(dic))
    f.close()

streamNet = gpd.read_file('../Output/streams.geojson')
streamNet.crs = {'init' :'epsg:32613'}
# The polygonize argument defaults to the grid mask when no arguments are supplied
shapes = grid.polygonize()

# Plot catchment boundaries
fig, ax = plt.subplots(figsize=(6.5, 6.5))

for shape in shapes:
    coords = np.asarray(shape[0]['coordinates'][0])
    ax.plot(coords[:,0], coords[:,1], color='cyan')

ax.set_xlim(grid.bbox[0], grid.bbox[2])
ax.set_ylim(grid.bbox[1], grid.bbox[3])
ax.set_title('Catchment boundary (vector)')
gpd.plotting.plot_dataframe(streamNet, None, cmap='Blues', ax=ax)
#ax = streamNet.plot()
mplleaflet.display(fig=ax.figure, crs=streamNet.crs, tiles='esri_aerial')