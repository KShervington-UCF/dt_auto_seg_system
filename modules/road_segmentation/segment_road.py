import os
import cv2
from sam2_model import SAM2Model

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

    print(f"Confidence score: {confidence_core}")
    print(f"Generated mask: {mask}")