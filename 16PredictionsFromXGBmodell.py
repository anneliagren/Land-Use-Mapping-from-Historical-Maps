#Predict maps for the XGB modell

import os
import numpy as np
from osgeo import gdal_array
import rasterio as rio
import pandas as pd
import xgboost as xgb
import pickle


# Load the trained XGBoost model'
model_file_path = '/workspace/data/XGBresults/100runsModell20/xgboost_model20.pkl'
with open(model_file_path, 'rb') as f:
    xgboost_model20 = pickle.load(f)


# Set the base directory
base_dir = '/workspace/data/AllMapsInOneFolder/decompressed/'
output_dir = '/workspace/data/AllMapsInOneFolder/predictions/'


# Define the correct order of bands ()
correct_order = [ 'Hue',  'B_G', 'B_R', 'G_R', 'GaussR10', 'MedianFR19', 'MinFG3', 'StDevFR19', 'StDevFI49', 'AverageG',  'AverageH', 'AverageS']

#Include these folders:
#'Class', 'Hue', 'B_G', 'B_R', 'G_R', 'GaussR10', 'MedianFR19', 'MinFG3', 'StDevFR19', 'StDevFI49', 'AverageG', 'AverageH', 'AverageS'

#Removed these folders:
#'Red', 'Green', 'Blue', 'Intensity', 'Saturation', 'AverageR', 'AverageB', 'AverageI', 'StDevFI11'


# Collect subfolder names
subfolder_names = [name for name in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, name))]

# Iterate over the files in the original raster composite folder
for original_tif_name in os.listdir(base_dir):
    bands_list = []
    if original_tif_name.endswith('.tif'):
        original_tif_path = os.path.join(base_dir, original_tif_name)
        
        # Get the size of the TIFF file
        with rio.open(original_tif_path) as src:
            tif_height, tif_width = src.height, src.width  
        for subfolder in os.listdir(base_dir):
                    subfolder_path = os.path.join(base_dir, subfolder)
                    if os.path.isdir(subfolder_path):
                        subfolder_file_path = os.path.join(subfolder_path, original_tif_name)
                        if os.path.exists(subfolder_file_path):
                            band = gdal_array.LoadFile(subfolder_file_path)
                            bands_list.append(band)
            

        # Make list into array
        bands_list = np.array(bands_list)
        # Reshape the array to match the input format during training
        bands_list_reshape = bands_list.reshape(12, tif_height * tif_width).T

        # Create a DataFrame for prediction
        xgb_data = pd.DataFrame(bands_list_reshape, columns=subfolder_names)
        xgb_data = xgb_data[correct_order]
        

        # Make a DMatrix
        d_data = xgb.DMatrix(xgb_data)

        # Make predictions
        predictions = xgboost_model20.predict(d_data)
        pred = predictions.reshape(tif_height, tif_width)

        # Create output folders for the predictions
        output_predictions_folder = os.path.join(output_dir, 'PredictedLandUse')
        # Ensure output folders exist
        os.makedirs(output_predictions_folder, exist_ok=True)

        output_file = os.path.join(output_predictions_folder, original_tif_name)
        # Save predictions to a new raster file
        with rio.open(original_tif_path) as reference_raster:
            profile = reference_raster.profile
            profile.update(dtype=rio.int16, count=1)

        with rio.open(output_file, 'w', **profile) as dst:
            dst.write(pred, 1)
            print('Done')