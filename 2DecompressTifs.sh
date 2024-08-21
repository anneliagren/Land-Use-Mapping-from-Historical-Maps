
#Unpack compressed tiff files
mkdir -p /workspace/data/AllMapsInOneFolder/decompressed

for file in /workspace/data/AllMapsInOneFolder/*.tif; do
    echo "Processing $file"
    gdal_translate -co COMPRESS=NONE "$file" "/workspace/data/AllMapsInOneFolder/decompressed/$(basename $file)"
done