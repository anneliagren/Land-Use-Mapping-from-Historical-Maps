#Georeference
import os
import fiona
import rasterio
from rasterio.transform import from_origin

image_file_folder = '/workspace/data/AllMapsNorrlandsInland/Cropped'
shp_path = '/workspace/data/Norrlands_inland/Norrlands_Inland_RT90.shp'
output_folder = '/workspace/data/AllMapsNorrlandsInland/Cropped/georeferenced'

# Ensure output directory exists
os.makedirs(output_folder, exist_ok=True)

# remove the 'J' in the beginning of Bladnummer and add '_0' to the end of the Bladnummer, 
#change a - to _ and change capitals to lower letters so the names match up. 
#It is taking the "Bladnummer" value from the second character to the end, converting it to lowercase, replacing the first '144-' with '144_', and then adding '_0' at the end.)

# Load the shapefile
with fiona.open(shp_path, "r") as shapefile:
    features = {feature["properties"]["Bladnummer"][1:].lower().replace('141-', '141_', 1) + '_0': feature["geometry"] for feature in shapefile}

# Loop over each TIFF file
for filename in os.listdir(image_file_folder):
    # Check if the file is a TIFF file
    if filename.lower().endswith('.tif'):
        # Remove the file extension from the filename
        name = os.path.splitext(filename)[0]

        # Find the corresponding geometry in the shapefile
        geom = features.get(name)
       
        if geom is not None:
            # Load the TIFF file
            with rasterio.open(os.path.join(image_file_folder, filename)) as src:
                data = src.read()

                # Get the x and y coordinates of the first point of the first polygon
                x, y = geom["coordinates"][0][0][0], geom["coordinates"][0][0][1]

                # Create the transform
                transform = from_origin(x, y, 2, 2)
                # Define the new CRS
                crs = "EPSG:3021"

                # Create a new TIFF file with the new transform and CRS in the output directory
                with rasterio.open(os.path.join(output_folder, f'geo_{filename}'), 'w', driver='GTiff', height=src.height, width=src.width, count=src.count, dtype=str(data.dtype), crs=crs, transform=transform) as dst:
                    dst.write(data)

            print(f'Georeferenced file saved: {os.path.join(output_folder, f"geo_{filename}")}')