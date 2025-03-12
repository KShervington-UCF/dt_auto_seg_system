import os
import cv2
import json
import numpy as np
from sam2_model import SAM2Model

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

    return [[road_point_x, road_point_y], [negative_point_x, negative_point_y]]

if __name__ == '__main__':
    # Load SAM2 model
    masking_model = SAM2Model()

    # Get path to image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, 'test_images', '1738856805878.jpg')

    image = cv2.imread(image_path)

    input_points = (calculate_input_points(image))

    mask, confidence_core = masking_model.segment_road(image_path, input_points=input_points, input_labels=([1, 0]))

    segmentation_poly = convert_mask_to_polygon(mask)

    output_dir = os.path.join(script_dir, 'output')
    os.makedirs(output_dir, exist_ok=True)

    json_data = {
        'image': os.path.basename(image_path),
        'class': 'TBD',
        'confidence_score': float(confidence_core),
        'segmentation': segmentation_poly
    }

    # Create json file to dump polygon data
    json_path = os.path.join(output_dir, 'polygon.json')
    with open(json_path, 'w') as f:
        json.dump(json_data, f)

    print(f"Output saved to {json_path}")