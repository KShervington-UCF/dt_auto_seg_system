# Digital Twin Auto Segmentation System

A pipeline for processing georeferenced imagery data and creating digital twin outputs following the TrainingDML (TDML) data standard.

## Project Requirements

- Python 3.9.18 for main environment ("env")
- Python ≥3.10 for SAM2 environment ("sam2-env")
- Python 3.9 for Detectron2 environment ("detectron2-env")

## Project Structure

The project has been restructured to follow a more standardized layout:

```
dt_auto_seg_system/
├── config.yaml             # Centralized configuration file
├── pipeline.py             # Main pipeline orchestration script
├── data/                   # Data directory
│   ├── raw/                # Raw input data
│   ├── processed/          # Intermediate processing outputs
│   └── output/             # Final pipeline outputs
├── modules/                # Pipeline modules
│   ├── preprocessing/      # Data preprocessing modules
│   │   ├── create_synced_df/
│   │   └── crop_images/
│   ├── geopose/            # Geopose creation module
│   ├── classification/     # Image classification module
│   ├── road_segmentation/  # Road segmentation using SAM2
│   ├── object_segmentation/ # Object segmentation using Detectron2
│   ├── tdml_integration/   # TDML format integration
│   └── utils/              # Shared utilities
└── docs/                   # Documentation
```

## Configuration System

This project uses a centralized configuration system with all paths and parameters defined in `config.yaml`. The configuration is loaded and managed by the `modules/utils/config_utils.py` module, which provides a singleton `Config` class for accessing settings across all modules.

Key features of the configuration system:
- Standardized paths for all input/output operations
- Automatic directory creation for outputs
- Path resolution relative to the project root

## Environment Setup

Refer to [Project Setup](./docs/PROJECT_SETUP.md) for detailed environment setup instructions.

The pipeline requires three separate virtual environments due to conflicting package dependencies:

1. **Main environment (env)** - For preprocessing, geopose, classification, and integration
2. **SAM2 environment (sam2-env)** - For road segmentation using SAM2
3. **Detectron2 environment (detectron2-env)** - For object segmentation using Detectron2

For the Detectron2 environment, follow the instructions [here](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to **Build Detectron2 from Source**. Specific package versions can be found in the corresponding `requirements.txt` file.

## Pipeline Execution

### Automated Pipeline Execution

The entire pipeline can now be executed with a single command using the orchestration script:

```bash
python pipeline.py
```

This will automatically:
1. Run each module in its appropriate virtual environment
2. Ensure data flows correctly between modules
3. Produce a final TDML output file

Options:
- `--skip-preprocessing` - Skip preprocessing steps (useful for reruns)

### Manual Pipeline Execution

For manual step-by-step testing, follow this process:

1. Place raw Sensor Logger data in the `data/raw/` directory
   - The data should include at least: `Camera/` directory, `Location.csv`, and `Orientation.csv`

2. Run each module in sequence with the appropriate environment:
   ```bash
   # In main environment (env)
   python modules/preprocessing/create_synced_df/create_synchronized_df.py
   python modules/preprocessing/crop_images/crop_images.py
   python modules/geopose/create_geopose.py
   python modules/classification/classify_images.py

   # In SAM2 environment (sam2-env)
   python modules/road_segmentation/segment_road.py

   # In Detectron2 environment (detectron2-env)
   python modules/object_segmentation/segment_objects.py

   # In main environment (env)
   python modules/tdml_integration/create_tdml.py
   ```

3. Final outputs will be available in:
   - Timestamped files in each module's output directory
   - Standardized paths in `data/processed/` and `data/output/`
   - Final TDML output at the path defined in `config.yaml`

## Data Flow

The pipeline processes data in the following sequence:

1. **Preprocessing**
   - Creates synchronized dataframe from raw sensor data
   - Crops images to focus on road surfaces

2. **Geopose Generation**
   - Creates geopose JSON from synchronized dataframe

3. **Image Classification**
   - Classifies cropped images

4. **Road Segmentation**
   - Generates road surface segmentation masks

5. **Object Segmentation**
   - Detects and segments objects in the scene

6. **TDML Integration**
   - Combines all outputs into a single TDML standard JSON file
