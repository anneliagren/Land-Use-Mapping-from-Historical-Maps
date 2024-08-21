
import os
os.environ["WBT_LINUX"] = "MUSL"
import whitebox
whitebox.download_wbt(linux_musl=True, reset=True)
wbt = whitebox.WhiteboxTools()



# # SetInput
folder_path_R = '/workspace/data/AllMapsInOneFolder/decompressed/Red'
folder_path = '/workspace/data/AllMapsInOneFolder/decompressed'


# # Create output folders for filtered tiff
output_GaussisanFRsigma10_folder = os.path.join(folder_path, 'GaussR10')


# # Ensure output folder exist

os.makedirs(output_GaussisanFRsigma10_folder, exist_ok=True)

# # Iterate over the files in the folder

for file_name in os.listdir(folder_path_R):
    file_path_R = os.path.join(folder_path_R, file_name)
    #check if the file is a TIFF
    if file_name.endswith('.tif'):
        base_name = os.path.splitext(file_name)[0]   
        
    
        # Construct the output file path for the filter
        output_GaussisanFRsigma10 = os.path.join(output_GaussisanFRsigma10_folder, f"{base_name}.tif")
            
  
        # Run ThiteboxTools FastAlmostGaussianFilter on Red band

        wbt.fast_almost_gaussian_filter(
        i=file_path_R, 
        output=output_GaussisanFRsigma10,
        sigma=10, 
        )
                
    
        print(f"Processing completed for {file_name}")





