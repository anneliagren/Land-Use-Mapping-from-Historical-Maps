import os
import shutil

# Set the source and destination directories
src_dir = '/workspace/data/All_Economic_Maps'
dst_dir = '/workspace/dataMAllMapsInOneFolder'

# Make sure the destination directory exists
os.makedirs(dst_dir, exist_ok=True)


# Walk through the source directory
for dirpath, dirnames, filenames in os.walk(src_dir):
    for file in filenames:
        if file.endswith('.tif'):
            # Construct the full file paths
            src_file = os.path.join(dirpath, file)
            dst_file = os.path.join(dst_dir, file)

            # Copy the file
            shutil.copy2(src_file, dst_file)

print('All TIFF files have been copied.')