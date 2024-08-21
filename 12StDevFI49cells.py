
import os
os.environ["WBT_LINUX"] = "MUSL"
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()

# # SetInput
folder_path_I = '/workspace/data/AllMapsInOneFolder/decompressed/Intensity'
folder_path = '/workspace/data/AllMapsInOneFolder/decompressed'

# # Create output folder
output_StDevFI49_folder = os.path.join(folder_path, 'StDevFI49')

# # Ensure output folder exist
os.makedirs(output_StDevFI49_folder, exist_ok=True)

# # Iterate over the files in the folder
for file_name in os.listdir(folder_path_I):
    file_path_I = os.path.join(folder_path_I, file_name)
    #check if the file is a TIFF
    if file_name.endswith('.tif'):
        base_name = os.path.splitext(file_name)[0]   
        
    
        # Construct the output file path for the filter
        output_StDevFI49 = os.path.join(output_StDevFI49_folder, f"{base_name}.tif")
            
  
        # Run ThiteboxTools StdDevFilter with a kernel of 49 by 49 cells on the Intensity
        wbt.standard_deviation_filter(
            i=file_path_I, 
            output=output_StDevFI49,
            filterx=49,
            filtery=49,
        )
                
    
        print(f"Processing completed for {file_name}")



