#This script removes the class 0 (graphics) from the raster files and replaces it with NoData value.
import os
import glob
from osgeo import gdal
import numpy as np

input_raster_path_1  = '/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/LandUse8bit/'

tiff_files = glob.glob(os.path.join(input_raster_path_1, '*.tif'))

# Check if raster files are found
if not tiff_files:
    print("No raster files found in the specified directory:", input_raster_path_1)
    exit()

# Specify the output directory
output_dir_1 = '/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/LandUse8bit/Nodata/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir_1):
    os.makedirs(output_dir_1)

# Loop over all raster files
for tiff_file in tiff_files:
    output_filename_1 = os.path.basename(tiff_file)
    output_path_1 = os.path.join(output_dir_1, output_filename_1)

    # Open the input raster file
    ds = gdal.Open(tiff_file)
    band = ds.GetRasterBand(1)

    # Read the raster data
    raster_data = band.ReadAsArray()

    # Convert class 0 to NoData
    raster_data[raster_data == 0] = band.GetNoDataValue()

    # Create the output raster file
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path_1, ds.RasterXSize, ds.RasterYSize, 1, band.DataType)
    out_band = out_ds.GetRasterBand(1)

    # Write the updated raster data to the output file
    out_band.WriteArray(raster_data)

    # Set the NoData value
    out_band.SetNoDataValue(band.GetNoDataValue())

    # Copy the georeferencing information
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    # Close the datasets
    ds = None
    out_ds = None



input_raster_path_2  = '/workspace/data/Anneli/burned_water/'

tiff_files = glob.glob(os.path.join(input_raster_path_2, '*.tif'))

# Check if raster files are found
if not tiff_files:
    print("No raster files found in the specified directory:", input_raster_path_2)
    exit()

# Specify the output directory
output_dir_2 = '/workspace/data/Anneli/burned_water/Nodata/'

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir_2):
    os.makedirs(output_dir_2)

# Loop over all raster files
for tiff_file in tiff_files:
    output_filename_2 = os.path.basename(tiff_file)
    output_path_2 = os.path.join(output_dir_2, output_filename_2)

    # Open the input raster file
    ds = gdal.Open(tiff_file)
    band = ds.GetRasterBand(1)

    # Read the raster data
    raster_data = band.ReadAsArray()

    # Convert class 0 to NoData
    raster_data[raster_data == 0] = band.GetNoDataValue()

    # Create the output raster file
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path_2, ds.RasterXSize, ds.RasterYSize, 1, band.DataType)
    out_band = out_ds.GetRasterBand(1)

    # Write the updated raster data to the output file
    out_band.WriteArray(raster_data)

    # Set the NoData value
    out_band.SetNoDataValue(band.GetNoDataValue())

    # Copy the georeferencing information
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    # Close the datasets
    ds = None
    out_ds = None