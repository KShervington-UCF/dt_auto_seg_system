# Object Segmentation Model Comparison

This directory contains code to compare different object segmentation models:
- Detectron2 Mask R-CNN (existing implementation)
- YOLOv11n-seg (smallest YOLO11 segmentation model)
- YOLOv11s-seg (small YOLO11 segmentation model)
- YOLOv11m-seg (medium YOLO11 segmentation model)

## Directory Structure

```
model_comparisons/
├── compare_models.py    # Main comparison script
├── test_images/         # Place test images here
├── output/              # Output directory (created when running comparison)
│   ├── results/         # JSON results and report
│   └── visualizations/  # Visualization images with segmentation masks
└── requirements.txt     # Additional requirements for model comparison
```

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Place test images in the `test_images` directory. The script works with `.jpg`, `.png`, `.jpeg`, `.bmp`, and `.tiff` formats.

## Running the Comparison

Run the comparison script:

```bash
python compare_models.py
```

This will:
1. Process each test image with all models
2. Generate visualizations with segmentation masks
3. Measure performance metrics (inference time, object count, confidence)
4. Create a report comparing all models
5. Generate performance comparison charts

## Output

The script generates:

1. **Visualization images** for each model and each test image
2. **JSON results** with detailed performance metrics
3. **Markdown report** summarizing the comparison findings
4. **Performance charts** comparing inference time, objects detected, and confidence scores

## Models

- **Detectron2**: Using the Mask R-CNN R_50_FPN_3x architecture from the existing implementation
- **YOLOv11n-seg**: Nano version of YOLOv11 with segmentation capability
- **YOLOv11s-seg**: Small version of YOLOv11 with segmentation capability
- **YOLOv11m-seg**: Medium version of YOLOv11 with segmentation capability

## Note

The first time you run the script, it will download the YOLO models which may take some time depending on your internet connection.
