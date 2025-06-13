#!/usr/bin/env python3
"""
Digital Twin Auto Segmentation System Pipeline

This script orchestrates the execution of the entire pipeline by:
1. Running each module in its appropriate virtual environment using subprocess
2. Ensuring data flows correctly between the modules
3. Producing a final TDML output file

Usage:
    python pipeline.py
"""

import os
import sys
import subprocess
import argparse
import datetime
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).resolve().parent
sys.path.append(str(project_root))

# Import the config utility
from modules.utils.config_utils import config


def run_in_environment(env_name, script_path):
    """Run a script in the specified conda environment
    
    Args:
        env_name: Name of the environment folder (e.g., 'env', 'sam2-env', 'detectron2-env')
        script_path: Path to the script to run
    """
    print(f"Running {os.path.basename(script_path)} in {env_name} environment...")
    
    # Use absolute paths
    script_path = os.path.abspath(script_path)
    env_path = os.path.join(project_root, env_name)
    
    try:
        # Run the script in the local conda environment with prefix path
        result = subprocess.run(
            ["conda", "run", "--prefix", env_path, "python", script_path],
            check=True,
            capture_output=True,
            text=True
        )
        # Print script output
        print(result.stdout)
        if result.stderr:
            print(f"Warnings/Errors: {result.stderr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_path}: {e}")
        print(f"Script output: {e.stdout}")
        print(f"Script errors: {e.stderr}")
        return False


def run_pipeline(skip_preprocessing=False):
    """Run the complete pipeline from preprocessing to TDML integration"""
    pipeline_start = datetime.datetime.now()
    print(f"Starting pipeline at {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Create necessary directories
    for dir_path in ['data/raw', 'data/processed', 'data/output']:
        os.makedirs(os.path.join(project_root, dir_path), exist_ok=True)
    
    # Define script paths
    preproc_sync_script = os.path.join(project_root, 'modules', 'preprocessing', 'create_synced_df', 'create_synchronized_df.py')
    preproc_crop_script = os.path.join(project_root, 'modules', 'preprocessing', 'crop_images', 'crop_images.py')
    geopose_script = os.path.join(project_root, 'modules', 'geopose', 'create_geopose.py')
    classification_script = os.path.join(project_root, 'modules', 'classification', 'classify_images.py')
    road_segmentation_script = os.path.join(project_root, 'modules', 'road_segmentation', 'segment_road.py')
    object_segmentation_script = os.path.join(project_root, 'modules', 'object_segmentation', 'segment_objects.py')
    tdml_integration_script = os.path.join(project_root, 'modules', 'tdml_integration', 'create_tdml.py')
    
    # Check for raw data
    raw_data_dir = config.get_path('raw_dir')
    camera_dir = os.path.join(raw_data_dir, 'Camera')
    location_file = os.path.join(raw_data_dir, 'Location.csv')
    orientation_file = os.path.join(raw_data_dir, 'Orientation.csv')
    
    if not skip_preprocessing:
        # Check if required raw data exists
        if not (os.path.exists(camera_dir) and os.path.exists(location_file) and os.path.exists(orientation_file)):
            print("Error: Raw data not found. Please ensure the following exist:")
            print(f"- Camera directory: {camera_dir}")
            print(f"- Location file: {location_file}")
            print(f"- Orientation file: {orientation_file}")
            return False
    
        # 1. Run preprocessing - Synchronized DataFrame (env)
        print("\n=== Step 1: Creating synchronized dataframe ===")
        if not run_in_environment("env", preproc_sync_script):
            print("Failed to create synchronized dataframe. Pipeline halted.")
            return False
        
        # 2. Run preprocessing - Crop Images (env)
        print("\n=== Step 2: Cropping images ===")
        if not run_in_environment("env", preproc_crop_script):
            print("Failed to crop images. Pipeline halted.")
            return False
        
        # 3. Run geopose module (env)
        print("\n=== Step 3: Creating geopose file ===")
        if not run_in_environment("env", geopose_script):
            print("Failed to create geopose file. Pipeline halted.")
            return False
        
        # 4. Run classification module (env)
        print("\n=== Step 4: Classifying images ===")
        if not run_in_environment("env", classification_script):
            print("Failed to classify images. Pipeline halted.")
            return False
    else:
        print("\nSkipping preprocessing steps (1-4)...")
    
    # 5. Run road segmentation module (sam2-env)
    print("\n=== Step 5: Running road segmentation ===")
    if not run_in_environment("sam2-env", road_segmentation_script):
        print("Failed to run road segmentation. Pipeline halted.")
        return False
    
    # 6. Run object segmentation module (detectron2-env)
    print("\n=== Step 6: Running object segmentation ===")
    if not run_in_environment("detectron2-env", object_segmentation_script):
        print("Failed to run object segmentation. Pipeline halted.")
        return False
    
    # 7. Run TDML integration module (env)
    print("\n=== Step 7: Creating TDML output ===")
    if not run_in_environment("env", tdml_integration_script):
        print("Failed to create TDML output. Pipeline halted.")
        return False
    
    pipeline_end = datetime.datetime.now()
    pipeline_duration = pipeline_end - pipeline_start
    
    print("\n=== Pipeline Execution Complete ===")
    print(f"Started at: {pipeline_start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Finished at: {pipeline_end.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {pipeline_duration}")
    print(f"Final TDML output: {config.get_path('tdml_integration', 'tdml_file_path')}")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the Digital Twin Auto Segmentation pipeline")
    parser.add_argument("--skip-preprocessing", action="store_true", help="Skip preprocessing steps (1-4)")
    args = parser.parse_args()
    
    run_pipeline(skip_preprocessing=args.skip_preprocessing)
