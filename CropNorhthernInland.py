import cv2
import numpy as np
import os

# Usage
image_file_folder = '/workspace/data/AllMapsNorrlandsInland/'
binary_folder = '/workspace/data/AllMapsNorrlandsInland/Binary/'
cropped_image_folder = '/workspace/data/AllMapsNorrlandsInland/Cropped/'

def find_most_white_region(binary_image, target_size=(5000, 5000)):
    max_white_pixels = 0
    max_white_region = None
    
    img_height, img_width = binary_image.shape[:2]
    
    for step in [200, 100, 20, 5]:
        for y in range(0, img_height - target_size[1] + 1, step):
            for x in range(0, img_width - target_size[0] + 1, step):
                region = binary_image[y:y+target_size[1], x:x+target_size[0]]
                white_pixels = np.sum(region == 255)  # Count white pixels
                
                if white_pixels > max_white_pixels:
                    max_white_pixels = white_pixels
                    max_white_region = (x, y)
    
    return max_white_region

def crop_map(original_image, binary_image, target_size=(5000, 5000)):
    # Find the region with the most white pixels in the binary image
    x, y = find_most_white_region(binary_image)
    
    # Crop the original map to the region defined by the white area in the binary image
    cropped_image = original_image[y:y+target_size[1], x:x+target_size[0]]
    
    # Resize the cropped image to target size if necessary
    cropped_image = cv2.resize(cropped_image, target_size[::-1])
    
    return cropped_image

def crop_all_maps_in_folder(input_folder, output_folder, binary_folder):
    # Create output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create binary folder if it doesn't exist
    if not os.path.exists(binary_folder):
        os.makedirs(binary_folder)
    
    # Get list of TIFF files in the input folder
    tiff_files = [f for f in os.listdir(input_folder) if f.endswith('.tif')]
    
    # Process each TIFF file
    for tiff_file in tiff_files:
        input_image_path = os.path.join(input_folder, tiff_file)
        output_image_path = os.path.join(output_folder, tiff_file)
        
        # Skip this file if it has already been processed
        if os.path.exists(output_image_path):
            print(f"Skipping {input_image_path} because output file already exists")
            continue
        
        try:
            # Read the original image
            original_image = cv2.imread(input_image_path)
            
            # Convert the original image to grayscale
            grayscale_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2GRAY)
            
            # Apply Otsu's thresholding to create a binary image
            _, binary_image = cv2.threshold(grayscale_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Save the binary image
            binary_filename = os.path.join(binary_folder, tiff_file)
            cv2.imwrite(binary_filename, binary_image)
            
            # Crop the original map to preserve white areas
            cropped_map = crop_map(original_image, binary_image)
            
            # Save the cropped map
            cv2.imwrite(output_image_path, cropped_map)
            print(f"Cropped image saved: {output_image_path}")
        except Exception as e:
            print(f"Error processing {input_image_path}: {e}")


crop_all_maps_in_folder(image_file_folder,cropped_image_folder, binary_folder)
