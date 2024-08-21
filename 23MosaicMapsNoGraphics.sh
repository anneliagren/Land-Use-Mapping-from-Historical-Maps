
# Set the pathsNorth
output_folder="/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/Nodata/MaskedN"
output_tiff_folder="/workspace/data/mosaics/ClassifiedNoGraphics/"

# Create a VRT file from all TIFFs in the subfolder
gdalbuildvrt "$output_folder/composite.vrt" "$output_folder"/*.tif

# Convert the VRT file to a GeoTIFF
gdal_translate -ot Byte "$output_folder/composite.vrt" "$output_tiff_folder/LandUseNoGraphicsNorth.tif"



# Set the pathsMid
output_folder="/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/Nodata/MaskedM"

# Create a VRT file from all TIFFs in the subfolder
gdalbuildvrt "$output_folder/composite.vrt" "$output_folder"/*.tif

# Convert the VRT file to a GeoTIFF
gdal_translate -ot Byte "$output_folder/composite.vrt" "$output_tiff_folder/LandUseNoGraphicsMid.tif"



# Set the pathsSouth
output_folder="/workspace/data/AllMapsInOneFolder/predictions/PredictedLandUse/generalized/Nodata/MaskedS"

# Create a VRT file from all TIFFs in the subfolder
gdalbuildvrt "$output_folder/composite.vrt" "$output_folder"/*.tif

# Convert the VRT file to a GeoTIFF
gdal_translate -ot Byte "$output_folder/composite.vrt" "$output_tiff_folder/LandUseNoGraphicsSouth.tif"



# Set the pathsNorthernInland
output_folder="workspace/data/Anneli/burned_water/Nodata/MaskedGraphics"

# Create a VRT file from all TIFFs in the subfolder
gdalbuildvrt "$output_folder/composite.vrt" "$output_folder"/*.tif

# Convert the VRT file to a GeoTIFF
gdal_translate -ot Byte "$output_folder/composite.vrt" "$output_tiff_folder/LandUseNoGraphicsNorthernInland.tif"
