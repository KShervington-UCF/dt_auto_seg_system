import os
import cv2
import json
import numpy as np
import sys
from pathlib import Path
from sam2_model import SAM2Model
from datetime import datetime

# Add the project root to the Python path to import the config utils
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the config utility
from modules.utils.config_utils import config

def convert_mask_to_polygon(binary_mask):
    # Ensure the mask is uint8
    binary_mask = binary_mask.astype(np.uint8)

    # Find external contours of the binary mask
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Try to smooth contours
    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

    polygons = []
    for cnt in contours:
        # Flatten the contour array and convert to list
        cnt = cnt.squeeze()
        if cnt.ndim < 2:
            continue  # Skip if not enough points
        # Convert to list of (x,y) coordinates
        polygon = cnt.flatten().tolist()
        polygons.append(polygon)

    # Get the polygon with the most points
    polygons.sort(key=lambda x: len(x), reverse=True)

    return polygons[0]

def calculate_input_points_hardcode(image):
    # Use the same points that worked well in Colab
    road_point_x, road_point_y = 540, 200  # Positive point (road)
    negative_point_x, negative_point_y = 540, 900  # Negative point (non-road)
    return [[road_point_x, road_point_y], [negative_point_x, negative_point_y]]

def calculate_input_points(image):
    """
    Calculate the road point in the image.
    
    Args:
        image: Input image as numpy array
    """
    height, width = image.shape[:2]

    # Calculate center point of the image
    center_x = width // 2
    center_y = height // 2
    
    # Calculate the vertical offset (25% below center, which is halfway to bottom)
    vertical_offset = height // 4
    
    # New center point with offset
    road_point_x = center_x
    road_point_y = center_y + vertical_offset

    # Negative mask points
    negative_point_x = center_x
    negative_point_y = center_y - vertical_offset

    # Points given in reverse order to resolve some issues with SAM2
    return [[negative_point_x, negative_point_y], [road_point_x, road_point_y]]

def process_images(classifications_file, masking_model):
    """
    Process multiple images based on classifications file
    
    Args:
        classifications_file: Path to JSON file containing image classifications
        masking_model: Initialized SAM2Model instance
    """
    # Read classifications file
    with open(classifications_file, 'r') as f:
        classifications = json.load(f)

    # Get directory paths from config
    images_dir = config.get_path('road_segmentation', 'images_dir')
    
    results = []
    
    for item in classifications:
        image_path = os.path.join(images_dir, item['image'])
        if not os.path.exists(image_path):
            print(f"Warning: Image {item['image']} not found, skipping...")
            continue

        # Process image
        image = cv2.imread(image_path)
        input_points = calculate_input_points(image)
        mask, confidence_score = masking_model.segment_road(
            image_path, 
            input_points=input_points, 
            input_labels=([1, 0])
        )
        
        segmentation_poly = convert_mask_to_polygon(mask)
        
        # Create result object
        result = {
            'image': item['image'],
            'class': item['predicted_class'],
            'confidence_score': float(confidence_score),
            'segmentation': segmentation_poly
        }
        results.append(result)
        print(f"Processed {item['image']}")

    return results

if __name__ == '__main__':
    # Load SAM2 model
    masking_model = SAM2Model()

    # Get paths from config
    classifications_path = config.get_path('classification', 'classification_file_path')
    
    # Process all images
    results = process_images(classifications_path, masking_model)

    # Save results
    output_dir = config.get_path('road_segmentation', 'output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save to timestamp-based file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f'segmentations_{timestamp}.json'
    output_path = os.path.join(output_dir, output_file)
    
    # Also save to standard path defined in config
    standard_path = config.get_path('road_segmentation', 'segmentation_file_path')
    os.makedirs(os.path.dirname(standard_path), exist_ok=True)
    
    # Save to both locations
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    with open(standard_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"All segmentations saved to:\n- {output_path}\n- {standard_path}")