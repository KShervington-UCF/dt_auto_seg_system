#!/usr/bin/env python3
"""
TDML Integration Module

This module combines outputs from all pipeline components:
- Geopose data
- Classification results
- Road segmentation
- Object segmentation

Into a single JSON file following the TrainingDML data standard.
"""

import os
import json
import sys
import datetime
import re
from pathlib import Path

# Add the project root to the Python path to import the config utils
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the config utility
from modules.utils.config_utils import config


def load_json_file(file_path):
    """Load a JSON file and return its contents"""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def extract_geopose_params(params_string):
    """
    Extracts the latitude, longitude, altitude, and quaternion values from a 'parameters' string
    formatted for GeoPose.

    Args:
        params_string (string): String containing GeoPose translation and rotation data.
        Ex. "translation=[28.5909266537322, -81.18892360486201, -4.721954339183867]&rotation=[0.3568742498642263, 0.5483800092396545, -0.4091981297088115, 0.6359850680845256]"

    Returns:
        dict: Dictionary with latitude, longitude, altitude, and quaternion components (qx, qy, qz, qw).
    """
    # Extract all floating point numbers from the string
    float_numbers = re.findall(r"[-+]?\d*\.\d+|\d+", params_string)

    # Convert extracted strings to float numbers
    float_numbers = [float(num) for num in float_numbers]

    # Assign the values to the corresponding labels
    # We expect the order: latitude, longitude, altitude, qx, qy, qz, qw
    if len(float_numbers) != 7:
        raise ValueError(f"Expected 7 floating point numbers, but got {len(float_numbers)}. Extracted values: {float_numbers}")

    latitude, longitude, altitude, qx, qy, qz, qw = float_numbers

    # Return them in a dictionary
    return {
        "latitude": latitude,
        "longitude": longitude,
        "altitude": altitude,
        "qx": qx,
        "qy": qy,
        "qz": qz,
        "qw": qw
    }


def merge_data():
    """Merge data from all modules into a single TDML structure following TrainingDML standard"""
    # Load data from each component
    geopose_path = config.get_path('geopose', 'geopose_file_path')
    classification_path = config.get_path('classification', 'classification_file_path')
    road_segmentation_path = config.get_path('road_segmentation', 'segmentation_file_path')
    object_segmentation_path = config.get_path('object_segmentation', 'segmentation_file_path')
    
    # Load data from files
    geopose_data = load_json_file(geopose_path)
    classification_data = load_json_file(classification_path)
    road_segmentation_data = load_json_file(road_segmentation_path)
    object_segmentation_data = load_json_file(object_segmentation_path)
    
    # Check if all data was loaded successfully
    if None in [geopose_data, classification_data, road_segmentation_data, object_segmentation_data]:
        print("Error: Failed to load one or more input files. Aborting TDML creation.")
        return None
    
    # Create image map for easier lookup
    classification_map = {item['image']: item for item in classification_data}
    road_segmentation_map = {item['image']: item for item in road_segmentation_data}
    object_segmentation_map = {}
    for image_id, data in object_segmentation_data.items():
        object_segmentation_map[image_id] = data
    
    # Create the base TDML structure
    tdml_data = {
        "type": "AI_AbstractTrainingDataset",
        "id": "dt_auto_seg_system",
        "name": "Digital Twin Auto Segmentation System",
        "description": "Training dataset created from digital twin auto segmentation pipeline",
        "version": "1.0",
        "createdTime": datetime.datetime.now().isoformat(),
        "data": []
    }
    
    # Process each image and combine with geopose data
    image_list = sorted(list(classification_map.keys()))
    
    for idx, image_id in enumerate(image_list):
        # Skip if image doesn't have all required data
        if (image_id not in road_segmentation_map) or (image_id not in object_segmentation_map):
            print(f"Warning: Image {image_id} is missing some data, skipping")
            continue
        
        # Get the corresponding frame from geopose data based on order
        if idx >= len(geopose_data['innerFrameSeries']):
            print(f"Warning: No matching geopose frame for image {image_id}, skipping")
            continue
            
        # Extract data for this image from all sources
        classification = classification_map[image_id]
        road_segmentation = road_segmentation_map[image_id]
        object_data = object_segmentation_map[image_id]
        
        # Get corresponding geopose frame
        geopose_frame = geopose_data['innerFrameSeries'][idx]
        
        # Extract geopose parameters (lat, lon, alt, quaternion)
        try:
            gp_parameters = extract_geopose_params(geopose_frame['parameters'])
        except ValueError as e:
            print(f"Warning: Error processing geopose parameters for image {image_id}: {e}")
            continue
        
        # Extract timestamp from image name (assuming image name is timestamp.jpg/png)
        timestamp = image_id.split('.')[0]  # Remove file extension
        
        # Create the integrated data object for this image
        frame_data = {
            "type": "AI_AbstractTrainingData",
            "id": f"frame_{idx}",
            "dataURL": [f"images/{image_id}"],
            "validTime": int(timestamp) if timestamp.isdigit() else None,
            "geopose": {
                "authority": geopose_frame.get('authority', 'WGS84'),
                "id": geopose_frame.get('id', f'frame_{idx}'),
                "position": {
                    "lat": gp_parameters['latitude'],
                    "lon": gp_parameters['longitude'],
                    "altitude": gp_parameters['altitude']
                },
                "quaternion": {
                    "x": gp_parameters['qx'],
                    "y": gp_parameters['qy'],
                    "z": gp_parameters['qz'],
                    "w": gp_parameters['qw']
                }
            },
            "labels": [
                {
                    "type": "AI_SceneLabel",
                    "class": classification['predicted_class'],
                    "confidence": classification['confidence_score']
                },
                {
                    "type": "AI_RoadLabel",
                    "confidence": road_segmentation['confidence_score'],
                    "segmentation": {
                        "type": "Polygon",
                        "coordinates": road_segmentation['segmentation']
                    }
                }
            ]
        }
        
        # Add object detections as labels
        for obj in object_data['objects']:
            frame_data['labels'].append({
                "type": "AI_ObjectLabel",
                "class": obj['class'],
                "confidence": obj['confidence'],
                "object": {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": obj['segmentation']
                    }
                }
            })
        
        tdml_data['data'].append(frame_data)
    
    # Add counts
    tdml_data['amountOfTrainingData'] = len(tdml_data['data'])
    
    return tdml_data


def create_tdml_file():
    """Create the TDML file by integrating all outputs"""
    # Merge data from all modules
    tdml_data = merge_data()
    if tdml_data is None:
        return
    
    # Create output directory if it doesn't exist
    output_dir = config.get_path('tdml_integration', 'output_dir')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save the TDML file with a timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"tdml_output_{timestamp}.json"
    output_path = os.path.join(output_dir, output_file)
    
    # Also save to the standard path defined in config
    standard_path = config.get_path('tdml_integration', 'tdml_file_path')
    os.makedirs(os.path.dirname(standard_path), exist_ok=True)
    
    # Save to both locations
    with open(output_path, 'w') as f:
        json.dump(tdml_data, f, indent=2)
        
    with open(standard_path, 'w') as f:
        json.dump(tdml_data, f, indent=2)
    
    print(f"TDML file created successfully with {tdml_data['amountOfTrainingData']} frames")
    print(f"Output saved to:\n- {output_path}\n- {standard_path}")
    
    return tdml_data


if __name__ == "__main__":
    create_tdml_file()