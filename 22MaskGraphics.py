
from osgeo import gdal
import os
import glob
import numpy as np

# Define a function to fill NoData values iteratively with adaptive search distance
def fill_nodata_adaptive(band):
    # Get dimensions of the band
    rows, cols = band.YSize, band.XSize
    
    # Create a mask to keep track of filled cells
    filled_mask = np.zeros((rows, cols), dtype=bool)
    
    # Specify the initial search distance
    max_search_dist = 3
    
    # Specify the factor to increase the search distance exponentially, the first 5 values of the search distance would be 3, 6, 12, 24, and 48 pixels
    search_dist_factor = 2
    
    # Loop until all NoData values are filled
    while True:
        # Copy the original band data
        original_data = band.ReadAsArray()
        
        # Print the number of NoData values before filling
        print("NoData values before filling: ", np.sum(original_data == band.GetNoDataValue()))
        
        # Create a copy of the mask for this iteration
        current_filled_mask = filled_mask.copy()
        
        # Fill NoData values using the current search distance
        gdal.FillNodata(targetBand=band, maskBand=None, maxSearchDist=max_search_dist, smoothingIterations=0)
        
        # Update the filled_mask with newly filled cells
        filled_mask = np.logical_or(filled_mask, band.ReadAsArray() != band.GetNoDataValue())
        
        # Print the number of NoData values after filling
        print("NoData values after filling: ", np.sum(band.ReadAsArray() == band.GetNoDataValue()))
        
        # Check if any new cells were filled in this iteration
        if np.array_equal(current_filled_mask, filled_mask):
            break  # Break the loop if no new cells were filled
        
        # Increase the search distance exponentially for the next iteration
        max_search_dist *= search_dist_factor

# Specify the in-/output directory
input_dir_1 = 'workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/LandUse8bit/Nodata'
output_dir_1 = 'workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/LandUse8bit/Nodata/MaskedGraphics/'

# Check if the output directory exists, if not, create it
if not os.path.exists(output_dir_1):
    os.makedirs(output_dir_1)

# Get a list of all TIFF files in the input directory
tiff_files = glob.glob(os.path.join(input_dir_1, '*.tif'))

prefix = "masked_"  # Replace with your actual prefix

# Loop over all raster files
for tiff_file in tiff_files:
    output_filename_1 = prefix + os.path.basename(tiff_file)
    output_path = os.path.join(output_dir_1, output_filename_1)

    # Open the input raster file in read-only mode
    ds = gdal.Open(tiff_file, gdal.GA_ReadOnly)

    # Create a new TIFF file for the output
    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(output_path, ds.RasterXSize, ds.RasterYSize, ds.RasterCount, ds.GetRasterBand(1).DataType)

    # Copy the geotransform and projection from the input dataset to the output dataset
    out_ds.SetGeoTransform(ds.GetGeoTransform())
    out_ds.SetProjection(ds.GetProjection())

    # Get the first band of the output dataset
    band = out_ds.GetRasterBand(1)

    # Set the NoData value
    band.SetNoDataValue(99)
    
    # Copy the band data from the input dataset to the output dataset
    band.WriteArray(ds.GetRasterBand(1).ReadAsArray())

    # Fill NoData values iteratively with adaptive search distance
    fill_nodata_adaptive(band)

    # Close the datasets
    ds = None
    out_ds = None