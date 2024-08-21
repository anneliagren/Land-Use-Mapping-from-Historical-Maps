
import os
os.environ["WBT_LINUX"] = "MUSL"
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()


# # SetInput and output folders
folder_path_B = '/workspace/data/AllMapsInOneFolder/decompressed/Blue'
folder_path_R = '/workspace/data/AllMapsInOneFolder/decompressed/Red'
folder_path = '/workspace/data/AllMapsInOneFolder/decompressed/'


# # Create output folders for the ratio
output_B_R_folder = os.path.join(folder_path, 'B_R')


# # Ensure output folders exist
os.makedirs(output_B_R_folder, exist_ok=True)


# # Iterate over the files in the folder
for file_name in os.listdir(folder_path_B):
    file_path_B = os.path.join(folder_path_B, file_name)
    if file_name.endswith('.tif'):
        base_name = os.path.splitext(file_name)[0]   
        file_path_R = os.path.join(folder_path_R, f"{base_name}.tif")
    

        # Construct the output file paths for the divison
        output_B_R = os.path.join(output_B_R_folder, f"{base_name}.tif")
            
  
         # Run whiteboxtools RGB to IHS
        wbt.divide(
                input1=file_path_B, 
                input2=file_path_R, 
                output=output_B_R, 
                )
                
    
        print(f"Processing completed for {file_name}")

