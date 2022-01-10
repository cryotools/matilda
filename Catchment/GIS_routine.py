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

working_directory = home + "/Seafile/Ana-Lena_Phillip/data/matilda/Catchment/"
input_DEM = home + "/Seafile/EBA-CA/Azamat_AvH/workflow/data/Jyrgalang/static/jyrgalang_dem_alos.tif"
RGI_files = home + "/Seafile/EBA-CA/Tianshan_data/GLIMS/13_rgi60_CentralAsia/13_rgi60_CentralAsia.shp"
ice_thickness_files = home + "/Seafile/EBA-CA/Tianshan_data/Ice_thickness-original/"

x, y = 78.921583, 42.653278 # pouring point
ele_bands, ele_zones = 20, 100

output_path = home + "/Seafile/EBA-CA/Azamat_AvH/workflow/data/Jyrgalang/static/catchment/"


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
grid = Grid.from_raster(input_DEM, data_name='dem')
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
plotFigure(demView, 'Elevation')
plt.show()

# save as clipped TIF
grid.to_raster(demView, output_path + "catchment_DEM.tif")

# Create shapefile and save it
shapes = grid.polygonize()

schema = {
    'geometry': 'Polygon',
    'properties': {'LABEL': 'float:16'}
}

with fiona.open(output_path + "catchment_shapefile.shp", 'w',
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

## Cutting all glaciers within the catchment from the RGI shapefile
rgi_shapefile = gpd.GeoDataFrame.from_file(RGI_files)
catchment_shapefile = gpd.GeoDataFrame.from_file(output_path + "catchment_shapefile.shp")

glaciers_catchment = gpd.overlay(rgi_shapefile, catchment_shapefile, how='intersection')
fig, ax = plt.subplots(1, 1)
base = catchment_shapefile.plot(color='white', edgecolor='black')
glaciers_catchment.plot(ax=base, column="RGIId", legend=True)
plt.show()


## Copy/get ice thickness files for each glacier
glaciers_catchment.to_file(driver = 'ESRI Shapefile', filename= output_path + "glaciers_in_catchment.shp")

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




# ZONAL STATISTICS ARE MISSING!
# Executed in QGIS using contour polygons (gdal) and zonal statistics.
# For zonal statistics in QGIS we need elevation zones as polygons not as lines. The code from the "Contour polygons" tool is:
# gdal_contour -p -amax ELEV_MAX -amin ELEV_MIN -b 1 -i 10.0 -f "ESRI Shapefile" /home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Jyrgalang/static/catchment/catchment_DEM.tif /home/phillip/Seafile/EBA-CA/Azamat_AvH/workflow/data/Jyrgalang/static/catchment/glacier_contours_polygon.shp

# --> for 10m elevation bands
# A tiny splinter polygon in elevation band 3680 had to be removed manually because it contained 0 ice and caused NaNs. --> FILTER!


##
# calculate the mean ice thickness per elevation band and write table
gis_thickness = "/Seafile/EBA-CA/Azamat_AvH/workflow/data/Jyrgalang/static/catchment/ice_thickness_profile.csv"
catchment_area = 252
elezone_interval = 100
def round_elezones(x, base=100):
    return base * round(x/base)

glacier_profile = pd.read_csv(home + gis_thickness)
glacier_profile.rename(columns={'ELEV_MAX':'Elevation'}, inplace=True)
glacier_profile["WE"] = glacier_profile["_mean"]*0.908*1000
glacier_profile = glacier_profile.drop(columns=["ID", "ELEV_MIN", "_mean"])
glacier_profile["Area"] = glacier_profile["Area"]/catchment_area

glacier_profile["EleZone"] = round_elezones(glacier_profile["Elevation"], base=elezone_interval)
glacier_profile = glacier_profile.sort_values(by='Elevation', ascending=True).reset_index(drop=True)

glacier_profile.to_csv(output_path + "ice_thickness_profile_final.csv", index=False)





##
# file = "/home/ana/Desktop/Thickness/contours.shp"
# output = "/home/ana/Desktop/Thickness/contours2.shp"
# gdf1 = gpd.read_file(file)
# gdf2 = gpd.read_file(output_shapefile)
#
#
# gdf = gpd.GeoDataFrame(pd.concat([gdf1, gdf2]))
# gdf.plot()
# plt.show()
## combine into elevation zones and create dataframe.