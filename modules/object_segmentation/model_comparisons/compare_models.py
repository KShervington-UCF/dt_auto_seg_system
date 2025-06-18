"""
Model Comparison for Object Instance Segmentation

This script evaluates multiple object instance segmentation models:
- Detectron2 Mask R-CNN
- YOLOv8n-seg
- YOLOv8s-seg
- YOLOv8m-seg

The script compares performance metrics and generates visualization outputs.
"""

import os
import sys
import json
import time
import cv2
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from tabulate import tabulate
import detectron2
from detectron2.data import MetadataCatalog

# Add parent directory to path for imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

# Import the detectron2 model implementation
from modules.object_segmentation.segment_objects import setup_detector, segment_image, process_segmentation_output, visualize_segmentation

# Import YOLO models
from ultralytics import YOLO

class ModelEvaluator:
    def __init__(self, test_images_dir, output_dir):
        """Initialize the model evaluator.

        Args:
            test_images_dir: Directory containing test images
            output_dir: Directory to save output results
        """
        self.test_images_dir = test_images_dir
        self.output_dir = output_dir
        self.results_dir = os.path.join(output_dir, "results")
        self.visualization_dir = os.path.join(output_dir, "visualizations")
        
        # Create output directories
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)
        
        # Initialize models dictionary
        self.models = {}
        
    def setup_models(self):
        """Set up all models for comparison."""
        print("Setting up models...")
        
        # Set up Detectron2
        print("Setting up Detectron2...")
        detectron2_predictor, detectron2_cfg = setup_detector()
        self.models["detectron2"] = {
            "predictor": detectron2_predictor,
            "cfg": detectron2_cfg,
            "type": "detectron2"
        }
        
        # Set up YOLO models
        yolo_models = ["yolo11n-seg", "yolo11s-seg", "yolo11m-seg", "yolo11l-seg", "yolo11x-seg"]
        for model_name in yolo_models:
            print(f"Setting up {model_name}...")
            try:
                model = YOLO(model_name)
                self.models[model_name] = {
                    "predictor": model,
                    "type": "yolo"
                }
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                
        print(f"Successfully loaded {len(self.models)} models")
    
    def run_evaluation(self):
        """Run evaluation on all test images with all models."""
        # Get list of test images
        image_files = [f for f in os.listdir(self.test_images_dir) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff'))]
        
        if not image_files:
            print(f"No images found in {self.test_images_dir}")
            return
        
        print(f"Found {len(image_files)} test images")
        
        # Results dictionary to store all evaluation metrics
        all_results = {
            "per_image": {},
            "summary": {}
        }
        
        # Process each image with each model
        for image_file in image_files:
            image_path = os.path.join(self.test_images_dir, image_file)
            print(f"\nProcessing {image_file}...")
            
            image_results = {}
            
            # Create directory for this image's visualizations
            image_vis_dir = os.path.join(self.visualization_dir, os.path.splitext(image_file)[0])
            os.makedirs(image_vis_dir, exist_ok=True)
            
            # Load image once
            original_image = cv2.imread(image_path)
            if original_image is None:
                print(f"Error loading image: {image_path}")
                continue
            
            # Run each model on this image
            for model_name, model_info in self.models.items():
                print(f"  Running {model_name}...")
                
                # Time the inference
                start_time = time.time()
                
                if model_info["type"] == "detectron2":
                    # Run Detectron2 model
                    predictor = model_info["predictor"]
                    cfg = model_info["cfg"]
                    _, outputs = segment_image(image_path, predictor, cfg)
                    metadata = detectron2.data.MetadataCatalog.get(cfg.DATASETS.TRAIN[0])
                    
                    # Visualize and save
                    vis_image = visualize_segmentation(original_image, outputs, metadata)
                    
                    # Process results for metrics
                    instances = outputs["instances"].to("cpu")
                    num_objects = len(instances)
                    confidence_scores = instances.scores.numpy().tolist() if num_objects > 0 else []
                    
                elif model_info["type"] == "yolo":
                    # Run YOLO model
                    predictor = model_info["predictor"]
                    results = predictor(image_path, task="segment")
                    
                    # Process results for metrics
                    num_objects = len(results[0].boxes)
                    confidence_scores = results[0].boxes.conf.cpu().numpy().tolist() if num_objects > 0 else []
                    
                    # Visualize and save
                    vis_image = results[0].plot()
                    vis_image = cv2.cvtColor(vis_image, cv2.COLOR_RGB2BGR)  # Convert YOLO RGB to OpenCV BGR
                
                # Calculate inference time
                inference_time = time.time() - start_time
                
                # Save visualization
                vis_path = os.path.join(image_vis_dir, f"{model_name}_{image_file}")
                cv2.imwrite(vis_path, vis_image)
                
                # Store results for this model
                image_results[model_name] = {
                    "inference_time": inference_time,
                    "num_objects_detected": num_objects,
                    "confidence_scores": confidence_scores,
                    "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0,
                    "visualization_path": vis_path
                }
            
            # Store results for this image
            all_results["per_image"][image_file] = image_results
        
        # Calculate summary statistics
        self._calculate_summary_metrics(all_results)
        
        # Save all results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_path = os.path.join(self.results_dir, f"comparison_results_{timestamp}.json")
        
        with open(results_path, 'w') as f:
            json.dump(all_results, f, indent=2)
            
        print(f"\nEvaluation complete. Results saved to {results_path}")
        
        # Generate report
        self._generate_report(all_results)
        
        return all_results
    
    def _calculate_summary_metrics(self, all_results):
        """Calculate summary metrics across all images."""
        summary = {}
        
        for model_name in self.models.keys():
            # Initialize summary stats
            total_inference_time = 0
            total_objects = 0
            all_confidence_scores = []
            image_count = 0
            
            # Aggregate stats across images
            for image_file, image_results in all_results["per_image"].items():
                if model_name in image_results:
                    model_result = image_results[model_name]
                    total_inference_time += model_result["inference_time"]
                    total_objects += model_result["num_objects_detected"]
                    all_confidence_scores.extend(model_result["confidence_scores"])
                    image_count += 1
            
            # Calculate averages
            avg_inference_time = total_inference_time / image_count if image_count > 0 else 0
            avg_objects_per_image = total_objects / image_count if image_count > 0 else 0
            avg_confidence = np.mean(all_confidence_scores) if all_confidence_scores else 0
            
            summary[model_name] = {
                "avg_inference_time": avg_inference_time,
                "avg_objects_per_image": avg_objects_per_image,
                "avg_confidence": avg_confidence,
                "total_objects_detected": total_objects,
                "processed_images": image_count
            }
        
        # Store summary in results
        all_results["summary"] = summary
    
    def _generate_report(self, all_results):
        """Generate a comparative report of model performance."""
        print("\nGenerating performance report...")
        
        # Create report data for tabulation
        report_data = []
        headers = ["Model", "Avg. Inference Time (s)", "Avg. Objects Detected", "Avg. Confidence"]
        
        for model_name, stats in all_results["summary"].items():
            row = [
                model_name,
                f"{stats['avg_inference_time']:.4f}",
                f"{stats['avg_objects_per_image']:.2f}",
                f"{stats['avg_confidence']:.4f}"
            ]
            report_data.append(row)
        
        # Sort by inference time (faster models first)
        report_data.sort(key=lambda x: float(x[1]))
        
        # Generate text report
        report_text = "# Model Performance Comparison Report\n\n"
        report_text += f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        report_text += f"Number of test images: {len(all_results['per_image'])}\n\n"
        report_text += "## Summary Results\n\n"
        report_text += tabulate(report_data, headers=headers, tablefmt="pipe") + "\n\n"
        
        # Add per-image section
        report_text += "## Per-Image Results\n\n"
        for image_file, image_results in all_results["per_image"].items():
            report_text += f"### {image_file}\n\n"
            
            image_data = []
            for model_name, stats in image_results.items():
                row = [
                    model_name,
                    f"{stats['inference_time']:.4f}",
                    stats['num_objects_detected'],
                    f"{stats['avg_confidence']:.4f}"
                ]
                image_data.append(row)
            
            # Sort by inference time
            image_data.sort(key=lambda x: float(x[1]))
            report_text += tabulate(image_data, headers=headers, tablefmt="pipe") + "\n\n"
        
        # Add visualization paths
        report_text += "## Visualization Paths\n\n"
        report_text += "Visualization images are saved in the following directories:\n\n"
        for image_file in all_results["per_image"].keys():
            image_name = os.path.splitext(image_file)[0]
            vis_dir = os.path.join(self.visualization_dir, image_name)
            report_text += f"- {image_file}: {vis_dir}\n"
        
        # Save report
        report_path = os.path.join(self.results_dir, "comparison_report.md")
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        # Generate performance comparison charts
        self._generate_charts(all_results)
        
        print(f"Report generated and saved to {report_path}")
    
    def _generate_charts(self, all_results):
        """Generate charts comparing model performance."""
        summary = all_results["summary"]
        models = list(summary.keys())
        
        # Set up the figure with subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # Chart 1: Inference Time Comparison
        inference_times = [summary[model]["avg_inference_time"] for model in models]
        axes[0].bar(models, inference_times)
        axes[0].set_title('Average Inference Time (seconds)')
        axes[0].set_ylabel('Time (s)')
        axes[0].set_xlabel('Model')
        axes[0].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Chart 2: Objects Detected Comparison
        objects_detected = [summary[model]["avg_objects_per_image"] for model in models]
        axes[1].bar(models, objects_detected)
        axes[1].set_title('Average Objects Detected per Image')
        axes[1].set_ylabel('Object Count')
        axes[1].set_xlabel('Model')
        axes[1].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Chart 3: Confidence Score Comparison
        confidence_scores = [summary[model]["avg_confidence"] for model in models]
        axes[2].bar(models, confidence_scores)
        axes[2].set_title('Average Confidence Score')
        axes[2].set_ylabel('Confidence Score')
        axes[2].set_xlabel('Model')
        axes[2].grid(axis='y', linestyle='--', alpha=0.7)
        
        # Adjust layout and save
        plt.tight_layout()
        chart_path = os.path.join(self.results_dir, "performance_comparison_charts.png")
        plt.savefig(chart_path)
        plt.close()
        
        print(f"Performance charts saved to {chart_path}")


def main():
    """Main execution function."""
    # Set up paths
    base_dir = Path(__file__).parent
    test_images_dir = os.path.join(base_dir, "test_images")
    output_dir = os.path.join(base_dir, "output")
    
    # Check if test images directory exists and has images
    if not os.path.exists(test_images_dir):
        print(f"Test images directory not found: {test_images_dir}")
        print("Please create the directory and add test images before running.")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(test_images_dir, output_dir)
    
    # Setup models
    evaluator.setup_models()
    
    # Run evaluation
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
