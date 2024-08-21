
import os
import rasterio
import numpy as np
from concurrent.futures import ThreadPoolExecutor

input_folder = '/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/'
output_folder = '/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/LandUse8bit/'

# Check if output directory exists, if not, create it
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

def convert_to_8_bit(filename):
    # Check if the file already exists in the output directory
    if os.path.exists(os.path.join(output_folder, filename)):
        print(f"File {filename} already exists in the output directory.")
        return

    try:
        print(filename)
        with rasterio.open(os.path.join(input_folder, filename)) as src:
            img = src.read(1)  # read the first band

            # Check if the image is already 8-bit
            if img.dtype != np.uint8:
                # Handle NoData values
                nodata_value = 99
                img[img == src.nodata] = nodata_value

                img = img.astype(np.uint8)

            # Update the metadata
            meta = src.meta
            meta.update(dtype=rasterio.uint8, nodata=nodata_value)

        # Write the image back to a new file, preserving the metadata
        with rasterio.open(os.path.join(output_folder, filename), 'w', **meta) as dst:
            dst.write(img, 1)  # write the image to the first band

        print('done')
        print('-----------------------------------')
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# Get a list of all .tif files in the input directory
filenames = [f for f in os.listdir(input_folder) if f.endswith(".tif")]

# Use a ThreadPoolExecutor to process multiple images simultaneously
with ThreadPoolExecutor(max_workers=60) as executor:
    executor.map(convert_to_8_bit, filenames)

print('All done')