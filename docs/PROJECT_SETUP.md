# Digital Twin Auto Segmentation System - Setup Guide

## System Requirements

Project originally run and executed on:
- Windows 10 ver. 10.0.19045 Build 19045
- CUDA Toolkit 11.2
- cuDNN 8.1.0
- Tensorflow 2.10.1
- Miniconda3 (conda ver. 24.7.1)

## Project Structure

The project is structured as follows:

```
dt_auto_seg_system/
├── config.yaml             # Centralized configuration file
├── pipeline.py             # Main pipeline orchestration script
├── data/                   # Data directory
│   ├── raw/                # Raw input data
│   ├── processed/          # Intermediate processing outputs
│   └── output/             # Final pipeline outputs
└── modules/                # Pipeline modules
```

## Environment Setup

The pipeline requires **three separate environments** due to conflicting dependencies. 
Follow the instructions below to set up each environment.

### 1. Main Environment Setup (`env`)

This environment is used for preprocessing, geopose generation, classification, and TDML integration.

1. Install Miniconda
   - Check miniconda installation by executing `conda info` in a command prompt
   - If this doesn't work, try launching a **Python Command Prompt** and executing `conda init` to initialize your command prompt with conda capabilities

2. Launch a command prompt in this project's root folder

3. Create the main virtual environment:
   ```bash
   conda create --name env python=3.9.18
   ```
   Or use a local environment:
   ```bash
   conda create --prefix .\env python=3.9.18
   ```

4. Activate the virtual environment:
   ```bash
   conda activate env
   ```
   Or for a local environment:
   ```bash
   conda activate .\env
   ```

5. (_Optional but recommended_) Enable GPU support:
   ```bash
   conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
   ```

6. Install helpful packages:
   ```bash
   conda install ipykernel pip
   ```
   - `ipykernel`: for running kernels from conda
   - `pip`: for installing packages from `requirements.txt`

7. Install project dependencies:
   ```bash
   pip install -r requirements.txt
   ```

8. (_Optional_) Verify GPU support:
   ```python
   import tensorflow as tf

   if len(tf.config.list_physical_devices('GPU')) > 0:
       print("GPU is Available!")
   else:
       raise Exception("No GPU available")
   ```

### 2. SAM2 Environment Setup (`sam2-env`)

This environment is used for road segmentation using SAM2.

Follow the same steps as the main environment setup with these differences:

1. Create a virtual environment with Python 3.10 or higher:
   ```bash
   conda create --name sam2-env python=3.10
   ```
   Or for a local environment:
   ```bash
   conda create --prefix .\sam2-env python=3.10
   ```

2. Activate the SAM2 environment:
   ```bash
   conda activate sam2-env
   ```
   Or for a local environment:
   ```bash
   conda activate .\sam2-env
   ```

3. Install dependencies from the SAM2 requirements file:
   ```bash
   pip install -r sam2-requirements.txt
   ```

### 3. Detectron2 Environment Setup (`detectron2-env`)

This environment is used for object segmentation using Detectron2.

1. Create a virtual environment with Python 3.9:
   ```bash
   conda create --name detectron2-env python=3.9
   ```
   Or for a local environment:
   ```bash
   conda create --prefix .\detectron2-env python=3.9
   ```

2. Activate the Detectron2 environment:
   ```bash
   conda activate detectron2-env
   ```
   Or for a local environment:
   ```bash
   conda activate .\detectron2-env
   ```

3. Install PyTorch (CUDA version):
   ```bash
   conda install -c pytorch pytorch torchvision
   ```

4. Follow the instructions to [Build Detectron2 from Source](https://detectron2.readthedocs.io/en/latest/tutorials/install.html)

5. Install additional dependencies:
   ```bash
   pip install -r detectron2-requirements.txt
   ```

## Configuration System

The project uses a centralized configuration system with all paths and parameters defined in `config.yaml` at the project root. This configuration is managed by the `modules/utils/config_utils.py` module.

### Key Configuration Components

1. **Raw Data Paths**: Define the location of raw sensor data files
2. **Processed Data Paths**: Define where intermediate outputs are saved
3. **Output Paths**: Define where final outputs are saved
4. **Module-specific Parameters**: Define parameters specific to each module

### Using the Configuration

To access configuration settings in your code:

```python
from modules.utils.config_utils import config

# Get a specific path
raw_data_path = config.get_path('raw_dir')

# Get a module-specific path
geopose_output = config.get_path('geopose', 'output_dir')
```

## Running the Pipeline

### Automated Execution

The entire pipeline can be executed with a single command:

```bash
python pipeline.py
```

This script will:
1. Run each module in the appropriate virtual environment
2. Handle data flow between modules
3. Generate the final TDML output

For more options:

```bash
python pipeline.py --help
```

### Manual Execution

If you prefer to run modules individually, follow the instructions in the main README.md file.

## Preparing Raw Data

Place raw Sensor Logger data in the `data/raw/` directory with the following structure:

```
data/raw/
├── Camera/        # Directory with camera images
├── Location.csv   # GPS location data
└── Orientation.csv # Device orientation data
```
