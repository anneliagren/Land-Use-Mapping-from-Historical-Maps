

import os
os.environ["WBT_LINUX"] = "MUSL"
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()


# # SetInput
folder_path = '/workspace/data/AllMapsInOneFolder/decompressed'


# # Create output folders for each color band
output_intensity_folder = os.path.join(folder_path, 'Intensity')
output_hue_folder = os.path.join(folder_path, 'Hue')
output_saturation_folder = os.path.join(folder_path, 'Saturation')


# # Ensure output folders exist
os.makedirs(output_intensity_folder, exist_ok=True)
os.makedirs(output_hue_folder, exist_ok=True)
os.makedirs(output_saturation_folder, exist_ok=True)


# # Iterate over the files in the folder

for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file is a TIFF
    if file_name.endswith('.tif'):
        base_name = os.path.splitext(file_name)[0]

        # Construct the output file paths for each color band
        output_intensity = os.path.join(output_intensity_folder, f"{base_name}.tif")
        output_hue = os.path.join(output_hue_folder, f"{base_name}.tif")
        output_saturation = os.path.join(output_saturation_folder, f"{base_name}.tif")

  
        # Run whiteboxtools RGB to IHS
        
        wbt.rgb_to_ihs(
            intensity=output_intensity,
            hue=output_hue,
            saturation=output_saturation,
            composite=file_path,
            )

        print(f"Processing completed for {file_name}")





