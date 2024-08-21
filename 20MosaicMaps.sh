
# Set the pathsNorth
output_folder="/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/North8bit"

# Create a VRT file from all TIFFs in the subfolder
gdalbuildvrt "$output_folder/composite.vrt" "$output_folder"/*.tif

# Convert the VRT file to a GeoTIFF
gdal_translate -ot Byte "$output_folder/composite.vrt" "$output_folder/LandUseNorth.tif"



# Set the pathsMid
output_folder="/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/Mid8bit"

# Create a VRT file from all TIFFs in the subfolder
gdalbuildvrt "$output_folder/composite.vrt" "$output_folder"/*.tif

# Convert the VRT file to a GeoTIFF
gdal_translate -ot Byte "$output_folder/composite.vrt" "$output_folder/LandUseMid.tif"



# Set the pathsSouth
output_folder="/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/South8bit"

# Create a VRT file from all TIFFs in the subfolder
gdalbuildvrt "$output_folder/composite.vrt" "$output_folder"/*.tif

# Convert the VRT file to a GeoTIFF
gdal_translate -ot Byte "$output_folder/composite.vrt" "$output_folder/LandUseSouth.tif"


# Set the pathsNorth
output_folder="Z:/EconomicMap/AllMapsNorrlandsInland/Cropped/georeferenced/predictions/PredictedLandUse/generalized/RT90_2.5_GonV"

# Create a VRT file from all TIFFs in the subfolder
gdalbuildvrt "$output_folder/composite.vrt" "$output_folder"/*.tif

# Convert the VRT file to a GeoTIFF
gdal_translate -ot Byte "$output_folder/composite.vrt" "$output_folder/LandUseNorthInland.tif"


