#!/usr/bin/env python3
"""
Visualization Module for Digital Twin Auto Segmentation System

This module provides functions for visualizing segmentation results by overlaying
polygons on images with semi-transparency.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon
import json
from datetime import datetime


def overlay_segmentation(image, segmentation_poly, color='red', alpha=0.5, 
                        figsize=(12, 8), dpi=100, show=True):
    """
    Overlay segmentation polygons on an image with semi-transparency

    Args:
        image: Input image as numpy array or path to image file
        segmentation_poly: Polygon coordinates as list of [x, y] points
        color: Color for the segmentation overlay
        alpha: Transparency level (0.0 to 1.0)
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the output figure
        show: Whether to display the figure

    Returns:
        fig, ax: Matplotlib figure and axes with the overlay
    """
    # Load image if a path is provided
    if isinstance(image, str):
        if os.path.exists(image):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"Image file not found: {image}")
    
    # Get image dimensions for coordinate flipping
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display the image
    ax.imshow(image)
    
    # Convert polygon to format needed by matplotlib and flip coordinates
    if isinstance(segmentation_poly, list) and len(segmentation_poly) > 0:
        # Create polygon patches
        patches = []
        
        # Handle multiple polygons or single polygon
        if isinstance(segmentation_poly[0], list) and isinstance(segmentation_poly[0][0], (list, tuple)):
            # Multiple polygons
            for poly in segmentation_poly:
                # Flip polygon coordinates
                flipped_poly = []
                for point in poly:
                    flipped_x = img_width - point[0]
                    flipped_y = img_height - point[1]
                    flipped_poly.append([flipped_x, flipped_y])
                patches.append(Polygon(flipped_poly, True))
        else:
            # Single polygon - flip the coordinates
            flipped_poly = []
            for point in segmentation_poly:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    flipped_x = img_width - point[0]
                    flipped_y = img_height - point[1]
                    flipped_poly.append([flipped_x, flipped_y])
            patches.append(Polygon(flipped_poly, True))
            
        p = PatchCollection(patches, alpha=alpha, color=color, linewidth=0)
        ax.add_collection(p)
    
    # Set axis properties
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    if show:
        plt.show()
        
    return fig, ax


def visualize_segmentation_results(results_file, image_dir, output_dir=None, 
                                  color='red', alpha=0.5, save=True, show=False, 
                                  figsize=(12, 8), dpi=100):
    """
    Visualize segmentation results from a JSON file containing segmentation data

    Args:
        results_file: Path to JSON file with segmentation results
        image_dir: Directory containing the original images
        output_dir: Directory to save visualization results
        color: Color for segmentation overlay
        alpha: Transparency level (0.0 to 1.0)
        save: Whether to save visualizations to output_dir
        show: Whether to display each visualization
        figsize: Figure size as (width, height)
        dpi: Dots per inch for output figures
        
    Returns:
        Dictionary mapping image filenames to visualization output paths
    """
    # Create output directory if needed
    if save:
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(results_file), 'visualizations')
        os.makedirs(output_dir, exist_ok=True)
    
    # Load results from JSON
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    output_paths = {}
    for item in results:
        image_name = item['image']
        image_path = os.path.join(image_dir, image_name)
        
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}")
            continue
        
        # Get segmentation polygon
        if 'segmentation' in item:
            segmentation_poly = item['segmentation']
        else:
            print(f"Warning: No segmentation data for {image_name}")
            continue
        
        # Create visualization
        fig, ax = overlay_segmentation(
            image_path, 
            segmentation_poly,
            color=color, 
            alpha=alpha, 
            figsize=figsize, 
            dpi=dpi, 
            show=show
        )
        
        # Save visualization
        if save:
            output_filename = f"vis_{os.path.splitext(image_name)[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            output_paths[image_name] = output_path
            print(f"Visualization saved: {output_path}")
    
    return output_paths


def overlay_multiple_segmentations(image, segmentation_polys_dict, alpha=0.5, 
                              figsize=(12, 8), dpi=100, show=True):
    """
    Overlay multiple segmentation polygons on an image with semi-transparency
    using different colors for different segmentation types

    Args:
        image: Input image as numpy array or path to image file
        segmentation_polys_dict: Dictionary mapping segmentation types to polygons and colors
                                 e.g., {'road': {'poly': poly_coords, 'color': 'red'},
                                        'object': {'poly': poly_coords, 'color': 'blue'}}
        alpha: Transparency level (0.0 to 1.0)
        figsize: Figure size as (width, height)
        dpi: Dots per inch for the output figure
        show: Whether to display the figure

    Returns:
        fig, ax: Matplotlib figure and axes with the overlay
    """
    # Load image if a path is provided
    if isinstance(image, str):
        if os.path.exists(image):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            raise FileNotFoundError(f"Image file not found: {image}")
    
    # Get image dimensions for coordinate flipping
    img_height = image.shape[0]
    img_width = image.shape[1]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Display the image
    ax.imshow(image)
    
    # Add each segmentation type with its own color
    for seg_type, seg_data in segmentation_polys_dict.items():
        poly = seg_data.get('poly')
        color = seg_data.get('color', 'red')
        label = seg_data.get('label', seg_type)
        
        if poly is None or len(poly) == 0:
            continue
            
        # Create polygon patches
        patches = []
        
        try:
            # Convert polygon data to numpy array for easier manipulation
            import numpy as np
            np_poly = np.array(poly)
            
            # Helper function to flip a set of coordinates
            def flip_coordinates(coords):
                flipped_coords = []
                
                # Handle different coordinate formats
                if len(coords) == 0:
                    return np.array(flipped_coords)
                    
                # Check if we have a flat list of coordinates [x1, y1, x2, y2, ...]
                if isinstance(coords, (list, np.ndarray)) and isinstance(coords[0], (int, float, np.integer, np.floating)):
                    # It's a flat list, process pairs
                    for i in range(0, len(coords), 2):
                        if i + 1 < len(coords):  # Make sure we have both x and y
                            flipped_x = img_width - coords[i]
                            flipped_y = img_height - coords[i + 1]
                            flipped_coords.append([flipped_x, flipped_y])
                    return np.array(flipped_coords)
                
                # Handle list of coordinate pairs [[x1,y1], [x2,y2], ...]
                for i in range(len(coords)):
                    if isinstance(coords[i], (list, tuple, np.ndarray)) and len(coords[i]) >= 2:
                        flipped_x = img_width - coords[i][0]
                        flipped_y = img_height - coords[i][1]
                        flipped_coords.append([flipped_x, flipped_y])
                return np.array(flipped_coords)
            
            # Check polygon format
            if len(np_poly.shape) == 1 and isinstance(poly[0], (int, float)):
                # We have a flat list of numbers - convert to pairs
                if len(np_poly) % 2 != 0:
                    print(f"Warning: Skipping improper polygon with odd number of coordinates")
                    continue
                # Reshape into pairs
                np_poly = np.array([(poly[i], poly[i+1]) for i in range(0, len(poly), 2)])
                # Flip coordinates
                np_poly = flip_coordinates(np_poly)
                patches.append(Polygon(np_poly))
                
            elif len(np_poly.shape) == 2 and np_poly.shape[1] == 2:
                # Correct format: array of [x,y] points for a single polygon
                # Flip coordinates
                np_poly = flip_coordinates(np_poly)
                patches.append(Polygon(np_poly))
                
            elif len(np_poly.shape) == 3 and np_poly.shape[2] == 2:
                # Multiple polygons, each a list of [x,y] points
                for p in np_poly:
                    # Flip coordinates
                    p_flipped = flip_coordinates(p)
                    patches.append(Polygon(p_flipped))
                    
            elif isinstance(poly[0], list):
                # Might be a list of polygons
                for p in poly:
                    if isinstance(p, list):
                        # Convert to numpy for validation
                        p_array = np.array(p)
                        if len(p_array.shape) == 2 and p_array.shape[1] == 2:
                            # Valid polygon points
                            p_flipped = flip_coordinates(p_array)
                            patches.append(Polygon(p_flipped))
                        elif len(p_array.shape) == 1 and len(p) % 2 == 0:
                            # Flat list of coordinates - reshape into pairs
                            p_array = np.array([(p[i], p[i+1]) for i in range(0, len(p), 2)])
                            # Flip coordinates
                            p_flipped = flip_coordinates(p_array)
                            patches.append(Polygon(p_flipped))
            else:
                print(f"Warning: Unrecognized polygon format for {seg_type} - skipping")
                continue
                
        except Exception as e:
            print(f"Error processing polygon for {seg_type}: {str(e)}")
            print(f"Polygon data sample: {poly[:10] if len(poly) > 10 else poly}")
            continue
        
        # Skip if no valid polygons were created
        if not patches:
            continue
            
        p = PatchCollection(patches, alpha=alpha, color=color, linewidth=0, label=label)
        ax.add_collection(p)
    
    # Add legend to show segmentation types
    ax.legend()
    
    # Set axis properties
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    
    if show:
        plt.show()
        
    return fig, ax


def visualize_batch(image_paths, segmentation_polys, output_dir=None, 
                   color='red', alpha=0.5, save=True, show=False, 
                   figsize=(12, 8), dpi=100):
    """
    Visualize a batch of images with their corresponding segmentation polygons

    Args:
        image_paths: List of paths to images
        segmentation_polys: List of segmentation polygons corresponding to images
        output_dir: Directory to save visualization results
        color: Color for segmentation overlay
        alpha: Transparency level (0.0 to 1.0)
        save: Whether to save visualizations to output_dir
        show: Whether to display each visualization
        figsize: Figure size as (width, height)
        dpi: Dots per inch for output figures
        
    Returns:
        Dictionary mapping image filenames to visualization output paths
    """
    if len(image_paths) != len(segmentation_polys):
        raise ValueError("Number of images and segmentation polygons must match")
    
    # Create output directory if needed
    if save and output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    output_paths = {}
    for i, (image_path, segmentation_poly) in enumerate(zip(image_paths, segmentation_polys)):
        # Create visualization
        fig, ax = overlay_segmentation(
            image_path, 
            segmentation_poly,
            color=color, 
            alpha=alpha, 
            figsize=figsize, 
            dpi=dpi, 
            show=show
        )
        
        # Save visualization
        if save and output_dir:
            image_name = os.path.basename(image_path)
            output_filename = f"vis_{os.path.splitext(image_name)[0]}.png"
            output_path = os.path.join(output_dir, output_filename)
            fig.savefig(output_path, bbox_inches='tight')
            plt.close(fig)
            output_paths[image_name] = output_path
            print(f"Visualization {i+1}/{len(image_paths)}: {output_path}")
    
    return output_paths


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize segmentation results')
    parser.add_argument('--results', required=True, help='Path to JSON file with segmentation results')
    parser.add_argument('--image-dir', required=True, help='Directory containing original images')
    parser.add_argument('--output-dir', help='Directory to save visualization results')
    parser.add_argument('--color', default='red', help='Color for segmentation overlay')
    parser.add_argument('--alpha', type=float, default=0.5, help='Transparency level (0.0 to 1.0)')
    parser.add_argument('--show', action='store_true', help='Display visualizations')
    parser.add_argument('--no-save', action='store_true', help='Do not save visualizations')
    
    args = parser.parse_args()
    
    visualize_segmentation_results(
        args.results,
        args.image_dir,
        args.output_dir,
        color=args.color,
        alpha=args.alpha,
        save=not args.no_save,
        show=args.show
    )
