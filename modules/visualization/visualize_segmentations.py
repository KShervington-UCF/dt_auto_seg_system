#!/usr/bin/env python3
"""
Segmentation Visualization Script for Digital Twin Auto Segmentation System

This script processes segmentation results and creates visualizations of the 
segmentation polygons overlaid on the original images.
"""

import os
import sys
import json
import argparse
import glob
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

# Import the visualization module and config utility
from modules.visualization.visualize import visualize_segmentation_results, overlay_multiple_segmentations
from modules.utils.config_utils import config


def visualize_road_segmentations():
    """
    Process and visualize road segmentation results
    """
    # Get paths from config
    raw_images_dir = config.get_path('road_segmentation', 'images_dir')  # Uses original images
    output_dir = config.get_path('output_dir')
    
    # Road segmentation output directory and files
    road_seg_dir = config.get_path('road_segmentation', 'output_dir')
    
    # Find the most recent segmentation file
    segmentation_files = glob.glob(os.path.join(road_seg_dir, 'segmentations_*.json'))
    
    if not segmentation_files:
        print("No segmentation files found.")
        return False
    
    # Use the most recent file
    latest_seg_file = max(segmentation_files, key=os.path.getmtime)
    print(f"Using latest segmentation file: {latest_seg_file}")
    
    # Create visualization output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    road_vis_dir = os.path.join(output_dir, 'visualizations', 'road_segmentations', timestamp)
    os.makedirs(road_vis_dir, exist_ok=True)
    
    # Visualize segmentations
    print(f"Creating road segmentation visualizations...")
    output_paths = visualize_segmentation_results(
        results_file=latest_seg_file,
        image_dir=raw_images_dir,
        output_dir=road_vis_dir,
        color='red',
        alpha=0.5,
        save=True,
        show=False
    )
    
    # Save output paths for reference
    paths_file = os.path.join(road_vis_dir, 'visualization_paths.json')
    with open(paths_file, 'w') as f:
        json.dump(output_paths, f, indent=2)
    
    print(f"Road segmentation visualizations created: {len(output_paths)}")
    print(f"Visualizations saved to: {road_vis_dir}")
    
    return True


def visualize_object_segmentations():
    """
    Process and visualize object segmentation results
    """
    # Get paths from config
    raw_images_dir = config.get_path('object_segmentation', 'images_dir')  # Uses original images
    output_dir = config.get_path('output_dir')
    
    # Object segmentation output directory and files
    object_seg_dir = config.get_path('object_segmentation', 'output_dir')
    
    # Find the most recent segmentation file
    segmentation_files = glob.glob(os.path.join(object_seg_dir, 'object_segmentations_*.json'))
    
    if not segmentation_files:
        print("No object segmentation files found.")
        return False
    
    # Use the most recent file
    latest_seg_file = max(segmentation_files, key=os.path.getmtime)
    print(f"Using latest object segmentation file: {latest_seg_file}")
    
    # Create visualization output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    object_vis_dir = os.path.join(output_dir, 'visualizations', 'object_segmentations', timestamp)
    os.makedirs(object_vis_dir, exist_ok=True)
    
    # Visualize segmentations
    print(f"Creating object segmentation visualizations...")
    output_paths = visualize_segmentation_results(
        results_file=latest_seg_file,
        image_dir=raw_images_dir,
        output_dir=object_vis_dir,
        color='blue',
        alpha=0.5,
        save=True,
        show=False
    )
    
    # Save output paths for reference
    paths_file = os.path.join(object_vis_dir, 'visualization_paths.json')
    with open(paths_file, 'w') as f:
        json.dump(output_paths, f, indent=2)
    
    print(f"Object segmentation visualizations created: {len(output_paths)}")
    print(f"Visualizations saved to: {object_vis_dir}")
    
    return True


def visualize_combined_segmentations():
    """
    Create combined visualizations with both road and object segmentations on the same images
    """
    # Get paths from config
    raw_images_dir = config.get_path('road_segmentation', 'images_dir')  # Original images
    output_dir = config.get_path('output_dir')
    
    # Get segmentation files
    road_seg_dir = config.get_path('road_segmentation', 'output_dir')
    object_seg_dir = config.get_path('object_segmentation', 'output_dir')
    
    # Find latest segmentation files
    road_seg_files = glob.glob(os.path.join(road_seg_dir, 'segmentations_*.json'))
    object_seg_files = glob.glob(os.path.join(object_seg_dir, 'segmentation_results_*.json'))
    
    if not road_seg_files or not object_seg_files:
        print("Missing segmentation files. Need both road and object segmentations.")
        return False
    
    latest_road_file = max(road_seg_files, key=os.path.getmtime)
    latest_object_file = max(object_seg_files, key=os.path.getmtime)
    
    print(f"Using road segmentation file: {latest_road_file}")
    print(f"Using object segmentation file: {latest_object_file}")
    
    # Load segmentation data
    with open(latest_road_file, 'r') as f:
        road_segmentations = json.load(f)
    
    with open(latest_object_file, 'r') as f:
        object_segmentations = json.load(f)
        
    # Debug: Print out structure of the files
    print(f"Road segmentation file type: {type(road_segmentations)}")
    if isinstance(road_segmentations, list) and len(road_segmentations) > 0:
        print(f"First road segmentation item: {road_segmentations[0].keys() if isinstance(road_segmentations[0], dict) else 'not a dict'}")
    
    print(f"Object segmentation file type: {type(object_segmentations)}")
    if isinstance(object_segmentations, dict):
        print(f"Object segmentation keys: {object_segmentations.keys()}")
    elif isinstance(object_segmentations, list) and len(object_segmentations) > 0:
        print(f"First object segmentation item: {object_segmentations[0].keys() if isinstance(object_segmentations[0], dict) else 'not a dict'}")
    
    # Create mapping of image names to segmentations
    road_seg_map = {item['image']: item['segmentation'] for item in road_segmentations}
    
    # Handle different object segmentation formats
    object_seg_map = {}
    if isinstance(object_segmentations, list):
        # Format where each item is an image with segmentations
        for item in object_segmentations:
            if isinstance(item, dict) and 'image' in item:
                if 'segmentations' in item:
                    object_seg_map[item['image']] = item['segmentations']
                elif 'segmentation' in item:
                    object_seg_map[item['image']] = [item['segmentation']]  # Wrap in list for consistency
    elif isinstance(object_segmentations, dict):
        # Format where the file is a dict with image names as keys
        for img_name, segmentations in object_segmentations.items():
            if isinstance(segmentations, list):
                object_seg_map[img_name] = segmentations
            else:
                object_seg_map[img_name] = [segmentations]  # Wrap in list for consistency
    
    # Get unique set of all images
    all_images = set(list(road_seg_map.keys()) + list(object_seg_map.keys()))
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_vis_dir = os.path.join(output_dir, 'visualizations', 'combined_segmentations', timestamp)
    os.makedirs(combined_vis_dir, exist_ok=True)
    
    # Create combined visualizations
    output_paths = {}
    for image_name in all_images:
        image_path = os.path.join(raw_images_dir, image_name)
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Build segmentation dictionary
        seg_dict = {}
        
        # Add road segmentation if available
        if image_name in road_seg_map:
            seg_dict['road'] = {
                'poly': road_seg_map[image_name],
                'color': 'red',
                'label': 'Road'
            }
        
        # Add object segmentations if available
        if image_name in object_seg_map:
            object_polys = object_seg_map[image_name]
            print(f"\nDebugging object segmentations for {image_name}:")
            print(f"Type: {type(object_polys)}")
            print(f"First few items: {str(object_polys)[:200]}..." if len(str(object_polys)) > 200 else object_polys)
            
            # Group objects by class if needed
            try:
                # Handle different formats of object segmentations
                if isinstance(object_polys, list):
                    # List of object segmentations
                    for obj in object_polys:
                        if isinstance(obj, dict):
                            # Extract class and segmentation
                            obj_class = obj.get('class', 'object')
                            obj_poly = obj.get('segmentation')
                            
                            if not obj_poly:
                                # Try other keys that might contain the segmentation
                                for key in ['polygon', 'points', 'coords', 'coordinates']:
                                    if key in obj:
                                        obj_poly = obj[key]
                                        break
                                        
                            if obj_poly:
                                if obj_class not in seg_dict:
                                    seg_dict[obj_class] = {
                                        'poly': [],
                                        'color': 'blue',  # Default color for objects
                                        'label': obj_class
                                    }
                                
                                # Add this polygon to the class collection
                                seg_dict[obj_class]['poly'].append(obj_poly)
                                print(f"Added polygon for class {obj_class}, format: {type(obj_poly)}")
                        else:
                            # Non-dictionary format, use as polygon directly
                            if 'object' not in seg_dict:
                                seg_dict['object'] = {
                                    'poly': [],
                                    'color': 'blue',
                                    'label': 'Object'
                                }
                            seg_dict['object']['poly'].append(obj)
                            
                elif isinstance(object_polys, dict):
                    # Dictionary format - could be class -> polygons mapping
                    for class_name, polys in object_polys.items():
                        if class_name not in seg_dict:
                            seg_dict[class_name] = {
                                'poly': [],
                                'color': 'blue',
                                'label': class_name
                            }
                            
                        # Add polygons for this class
                        if isinstance(polys, list):
                            seg_dict[class_name]['poly'].extend(polys)
                        else:
                            seg_dict[class_name]['poly'].append(polys)
            except Exception as e:
                print(f"Error processing object segmentations for {image_name}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        # Create visualization with multiple segmentations
        fig, ax = overlay_multiple_segmentations(
            image_path,
            seg_dict,
            alpha=0.5,
            figsize=(12, 8),
            dpi=100,
            show=False
        )
        
        # Save visualization
        output_filename = f"combined_vis_{os.path.splitext(image_name)[0]}.png"
        output_path = os.path.join(combined_vis_dir, output_filename)
        fig.savefig(output_path, bbox_inches='tight')
        plt.close(fig)
        output_paths[image_name] = output_path
        print(f"Created combined visualization for {image_name}")
    
    # Save output paths for reference
    paths_file = os.path.join(combined_vis_dir, 'visualization_paths.json')
    with open(paths_file, 'w') as f:
        json.dump(output_paths, f, indent=2)
    
    print(f"Combined visualizations created: {len(output_paths)}")
    print(f"Visualizations saved to: {combined_vis_dir}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create visualizations of segmentation results")
    parser.add_argument("--type", choices=["road", "object", "combined", "all"], default="combined", 
                       help="Type of segmentation to visualize")
    args = parser.parse_args()
    
    if args.type == "road" or args.type == "all":
        visualize_road_segmentations()
    
    if args.type == "object" or args.type == "all":
        visualize_object_segmentations()
        
    if args.type == "combined" or args.type == "all":
        visualize_combined_segmentations()
