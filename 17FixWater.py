import os
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
import pandas as pd
from shapely.geometry import box  

# Get the list of water shapefiles
water_files = [os.path.join('/workspace/data/Water', f) for f in os.listdir('/workspace/data/Water') if f.endswith(".shp")]

# Load the water shapefiles and concatenate them into a single GeoDataFrame
water_gdfs = [gpd.read_file(shp) for shp in water_files]
water = pd.concat(water_gdfs, ignore_index=True)

# Get the list of raster files
raster_files = [os.path.join('/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/', f) for f in os.listdir('/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/') if f.endswith(".tif")]

for raster_file in raster_files:
    print(f"Processing {raster_file}") 
    with rasterio.open(raster_file) as src:
        # Read the original raster data
        original_data = src.read(1)

        # Create a bounding box around the raster
        raster_bbox = box(*src.bounds)

        # Select the water features that intersect with the bounding box
        water_to_burn = water[water.geometry.intersects(raster_bbox)]
        
        # Print the geometry of the water features
        print(water_to_burn.geometry)

        # Check if water_to_burn is empty
        if water_to_burn.empty:
            print(f"No water features found for {raster_file}")
            continue

        # Burn the selected water features into a separate raster
        burned = rasterize(
            ((geom, 4) for geom in water_to_burn.geometry),
            out_shape=src.shape,
            transform=src.transform,
            fill=0,
            all_touched=True,
        )
        # Print the output of the rasterize function
        print(burned)

        # Use the burned raster to modify the original raster data
        original_data[burned == 4] = 4

        # Define the new directory
        new_directory = "/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/burned_water"

        # Make sure the new directory exists
        os.makedirs(new_directory, exist_ok=True)

        # Get the filename from the raster file path
        filename = os.path.basename(raster_file)

        # Join the new directory and the filename to get the new file path
        new_file_path = os.path.join(new_directory, filename)

        # Write the modified raster to a new file
        with rasterio.open(new_file_path, "w", driver="GTiff", height=src.height, width=src.width, count=1, dtype=rasterio.uint8, crs=src.crs, transform=src.transform) as dst:
            dst.write(original_data, 1)