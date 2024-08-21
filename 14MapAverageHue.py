import os
os.environ["WBT_LINUX"] = "MUSL"
import numpy as np
import rasterio


# # SetInput
folder_path_Hue = '/workspace/data/AllMapsInOneFolder/decompressed/Hue/'
folder_path = '/workspace/data/AllMapsInOneFolder/decompressed'

# Create output folder
output_AverageHue_folder = os.path.join(folder_path, 'AverageH')

# Ensure output folder exist
os.makedirs(output_AverageHue_folder, exist_ok=True)

# Use os.listdir to get all files in the directory
files_in_directory = os.listdir(folder_path_Hue)

# Filter out the files to only have files with .tif extension
tif_files = [file for file in files_in_directory if file.endswith('.tif')]

# Now you can iterate over the tif files
for tif_file in tif_files:
    file_path = os.path.join(folder_path_Hue, tif_file)
    # Open the tif file and convert it to numpy array
    with rasterio.open(file_path) as src:
        img_array = src.read(1)

        print(f"Processing file: {file_path}")
        print(f"Original shape: {img_array.shape}")

        # Calculate the average of the numpy array
        average = np.average(img_array)

        # Create a new numpy array with the same shape as the original but filled with the average value
        average_array = np.full(img_array.shape, average)

        # Copy the metadata from the source file
        metadata = src.meta.copy()

        # Update the metadata to reflect the number of bands and the data type
        metadata.update({
            'count': 1,
            'dtype': str(average_array.dtype),
            'crs': 'EPSG:3021'
        })

        # Write the average array out as a new tif file
        output_file_path = os.path.join(output_AverageHue_folder, tif_file)
        with rasterio.open(output_file_path, 'w', **metadata) as dst:
            dst.write(average_array, 1)

        print(f"Output file written to: {output_file_path}")