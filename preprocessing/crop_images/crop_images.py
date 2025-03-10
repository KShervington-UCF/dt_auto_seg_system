#!/usr/bin/env python3
# A script which crops images for classification

import os
import cv2
import numpy as np
from tqdm import tqdm

def crop_image(image, crop_size=(224, 224)):
    """
    Crop a 224x224 slice from the image with the center point offset vertically
    by 25% below the image center (halfway between center and bottom edge).
    
    Args:
        image: Input image as numpy array
        crop_size: Size of the crop (width, height)
    
    Returns:
        Cropped image as numpy array
    """
    height, width = image.shape[:2]
    
    # Calculate center point of the image
    center_x = width // 2
    center_y = height // 2
    
    # Calculate the vertical offset (25% below center, which is halfway to bottom)
    vertical_offset = height // 4
    
    # New center point with offset
    new_center_x = center_x
    new_center_y = center_y + vertical_offset
    
    # Calculate crop coordinates
    half_crop_width = crop_size[0] // 2
    half_crop_height = crop_size[1] // 2
    
    # Calculate crop boundaries
    left = max(0, new_center_x - half_crop_width)
    top = max(0, new_center_y - half_crop_height)
    right = min(width, new_center_x + half_crop_width)
    bottom = min(height, new_center_y + half_crop_height)
    
    # Crop the image
    cropped_image = image[top:bottom, left:right]
    
    # Resize if necessary to ensure exact dimensions
    if cropped_image.shape[0] != crop_size[1] or cropped_image.shape[1] != crop_size[0]:
        cropped_image = cv2.resize(cropped_image, crop_size)
    
    return cropped_image

def process_directory(input_dir, output_dir, crop_size=(224, 224)):
    """
    Process all images in the input directory and save cropped versions to the output directory.
    
    Args:
        input_dir: Directory containing source images
        output_dir: Directory to save cropped images
        crop_size: Size of the crop (width, height)
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Processing {len(image_files)} images...")
    
    # Process each image
    for filename in tqdm(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)
        
        try:
            # Read image
            image = cv2.imread(input_path)
            
            if image is None:
                print(f"Failed to read image: {input_path}")
                continue
            
            # Crop image
            cropped_image = crop_image(image, crop_size)
            
            # Save cropped image
            cv2.imwrite(output_path, cropped_image)
            
        except Exception as e:
            print(f"Error processing {input_path}: {e}")

def main():
    input_dir = os.path.join(os.path.dirname(__file__), '..', 'create_synced_df', 'sample_raw_data', 'Camera')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    crop_size = (224, 224)
    
    process_directory(input_dir, output_dir, crop_size)
    
    print("Image cropping completed!")

if __name__ == "__main__":
    main()

# Script created with Claude 3.7 Sonnet
# 
# Prompt:
# Create a script, by editing @crop_images.py, which takes the images in a specified directory and crops a 224 by 224 slice from each image and saves the cropped images in a specified directory. Each source image is taken from a camera mounted on the hood of a car, pointed forward. The intention of the crop is to get an image of only the road surface. Therefore, the crop should be taken from a specific location relative to the resolution of the image. The center point of the crop should be vertically offset below the center point of the image by 25%. To be clear, this point should be halfway between the center of the image and the bottom edge of the image.
