# Start WhiteboxTools
import os
os.environ["WBT_LINUX"] = "MUSL"
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()

# # SetInput


folder_path = '/workspace/data/AllMapsInOneFolder/decompressed'
 #folder path and subfolder path Iterate over subfolders 

# # Create output folders for each color band

output_red_folder = os.path.join(folder_path, 'Red')
output_green_folder = os.path.join(folder_path, 'Green')
output_blue_folder = os.path.join(folder_path, 'Blue')


# # Ensure output folders exist

os.makedirs(output_red_folder, exist_ok=True)
os.makedirs(output_green_folder, exist_ok=True)
os.makedirs(output_blue_folder, exist_ok=True)


# # Iterate over the files in the folder


for file_name in os.listdir(folder_path):
    file_path = os.path.join(folder_path, file_name)
    
    # Check if the file is a TIFF
    if file_name.endswith('.tif'):
        base_name = os.path.splitext(file_name)[0]

        # Construct the output file paths for each color band
        output_R = os.path.join(output_red_folder, f"{base_name}.tif")
        output_G = os.path.join(output_green_folder, f"{base_name}.tif")
        output_B = os.path.join(output_blue_folder, f"{base_name}.tif")

        # Run whiteboxtools split_colour_composite for each color band
        wbt.split_colour_composite(
            i = file_path,
            red=output_R,
            green=output_G,
            blue=output_B
        )

        print(f"Processing completed for {file_name}")





