import os
os.environ["WBT_LINUX"] = "MUSL"
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()

# # SetInput
folder_path_G = '/workspace/data/AllMapsInOneFolder/decompressed/Green'
folder_path = '/workspace/data/AllMapsInOneFolder/decompressed/'


# # Create output folder
output_MinFG3_folder = os.path.join(folder_path, 'MinFG3')

# # Ensure output folder exist
os.makedirs(output_MinFG3_folder, exist_ok=True)


# # Iterate over the files in the folder
for file_name in os.listdir(folder_path_G):
    file_path_G = os.path.join(folder_path_G, file_name)
    #check if the file is a TIFF
    if file_name.endswith('.tif'):
        base_name = os.path.splitext(file_name)[0]   
        
    
        # Construct the output file path for the filter
        output_MinFG3 = os.path.join(output_MinFG3_folder, f"{base_name}.tif")
            
  
        # Run ThiteboxTools minimumfilter on the green band
        wbt.minimum_filter(
            i=file_path_G, 
            output=output_MinFG3,
            filterx=3,
            filtery=3,
        )
                
    
        print(f"Processing completed for {file_name}")


