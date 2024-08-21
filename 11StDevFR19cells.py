
import os
os.environ["WBT_LINUX"] = "MUSL"
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()

# # SetInput
folder_path_R = '/workspace/data/AllMapsInOneFolder/decompressed/Red'
folder_path = '/workspace/data/AllMapsInOneFolder/decompressed'

# # Create output folder
output_StDevFR19_folder = os.path.join(folder_path, 'StDevFR19')

# # Ensure output folder exist
os.makedirs(output_StDevFR19_folder, exist_ok=True)

# # Iterate over the files in the folder
for file_name in os.listdir(folder_path_R):
    file_path_R = os.path.join(folder_path_R, file_name)
    #check if the file is a TIFF
    if file_name.endswith('.tif'):
        base_name = os.path.splitext(file_name)[0]   
        
    
        # Construct the output file path for the filter
        output_StDevFR19 = os.path.join(output_StDevFR19_folder, f"{base_name}.tif")
            
  
        # Run ThiteboxTools StdDevFilter with a kernel of 19 by 19 cells on the red band
        wbt.standard_deviation_filter(
            i=file_path_R, 
            output=output_StDevFR19,
            filterx=19,
            filtery=19,
        )
                
    
        print(f"Processing completed for {file_name}")



