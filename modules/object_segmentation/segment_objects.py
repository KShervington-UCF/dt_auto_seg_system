import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
import os
import json
import sys
import datetime
import numpy as np
from pathlib import Path

# Add the project root to the Python path to import the config utils
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the config utility
from modules.utils.config_utils import config


def setup_detector():
    """Set up the Detectron2 model and return a predictor."""
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Set the confidence threshold
    predictor = DefaultPredictor(cfg)
    return predictor, cfg


def segment_image(image_path, predictor, cfg):
    """Segment objects in an image and return the output data."""
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error loading image: {image_path}")
        return None
    
    # Run prediction
    outputs = predictor(image)
    
    return image, outputs


def process_segmentation_output(outputs, metadata):
    """Process detectron2 segmentation output into a structured format."""
    instances = outputs["instances"].to("cpu")
    
    # Extract needed data
    masks = instances.pred_masks.numpy()
    classes = instances.pred_classes.numpy()
    scores = instances.scores.numpy()
    
    # Get class names
    class_names = [metadata.thing_classes[class_id] for class_id in classes]
    
    # Process each detected object
    objects = []
    for i in range(len(masks)):
        # Convert binary mask to polygon representation for more compact storage
        mask = masks[i].astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [contour.reshape(-1, 2).tolist() for contour in contours]
        
        object_data = {
            "class": class_names[i],
            "confidence": float(scores[i]),  # Convert numpy float to Python float for JSON serialization
            "segmentation": contours
        }
        objects.append(object_data)
    
    return objects


def visualize_segmentation(image, outputs, metadata):
    """Create a visualization of the segmentation results."""
    v = Visualizer(image[:, :, ::-1], metadata, scale=1.2)
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    return out.get_image()[:, :, ::-1]  # Convert back to BGR for OpenCV


def segment_objects(visualize=False):
    """Process all images in the input directory and save segmentation results."""
    # Set up paths from config
    input_dir = config.get_path('object_segmentation', 'images_dir')
    output_dir = config.get_path('object_segmentation', 'output_dir')
    vis_dir = os.path.join(output_dir, "segmented_images")
    
    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    if visualize:
        os.makedirs(vis_dir, exist_ok=True)
    
    # Set up the segmentation model
    predictor, cfg = setup_detector()
    metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
    
    # Initialize results structure
    all_results = {}
    
    # Process all images in the input directory
    image_files = [f for f in os.listdir(input_dir) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
    
    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)
        print(f"Processing {image_file}...")
        
        # Segment the image
        result = segment_image(image_path, predictor, cfg)
        if result is None:
            continue
            
        image, outputs = result
        
        # Process segmentation outputs
        objects = process_segmentation_output(outputs, metadata)
        
        # Save to results collection
        all_results[image_file] = {
            "image": image_file,
            "objects": objects
        }
        
        # Generate and save visualization if requested
        if visualize:
            vis_image = visualize_segmentation(image, outputs, metadata)
            vis_path = os.path.join(vis_dir, f"segmented_{image_file}")
            cv2.imwrite(vis_path, vis_image)
    
    # Save results to both the timestamp file and standard path
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = os.path.join(output_dir, f"segmentation_results_{timestamp}.json")
    
    # Also save to standard path defined in config
    standard_path = config.get_path('object_segmentation', 'segmentation_file_path')
    os.makedirs(os.path.dirname(standard_path), exist_ok=True)
    
    # Save to both locations
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
        
    with open(standard_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Segmentation complete. Results saved to:\n- {results_path}\n- {standard_path}")
    return all_results


if __name__ == "__main__":
    segment_objects(visualize=False)

