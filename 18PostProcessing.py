import os
import whitebox

wbt = whitebox.WhiteboxTools()

# Set the path to the WhiteboxTools executable (this is my local version of Whitebox Tools Extension on my PC with the extesion activated)
wbt.set_whitebox_dir(r'C:\Users\anag\AppData\Local\anaconda3\Lib\site-packages\whitebox')


# # SetInput

folder_path_classified_map = 'Z:/EconomicMap/AllMapsInOneFolder/predictions/PredictedLandUse/burned_water'
folder_path = 'Z:/EconomicMap/AllMapsInOneFolder/predictions/PredictedLandUse/'

# # Create output folder
folder_path_generalized_map = os.path.join(folder_path, 'generalized')


# # Ensure output folder exist
os.makedirs(folder_path_generalized_map, exist_ok=True)


# # Iterate over the files in the folder

for file_name in os.listdir(folder_path_classified_map):
    file_path_classified_map = os.path.join(folder_path_classified_map, file_name)
    #check if the file is a TIFF
    if file_name.endswith('.tif'):
        base_name = os.path.splitext(file_name)[0]   
        
    
        # Construct the output file path for the filter
        genralized_file = os.path.join(folder_path_generalized_map, f"{base_name}.tif")
            
  
        # Run ThiteboxTools Extension "Generalize Classified Raster" 
        wbt.generalize_classified_raster(
            i=file_path_classified_map, 
            output=genralized_file, 
            min_size=9, 
            method="longest", 
        )

        print(f"Processing completed for {file_name}")

