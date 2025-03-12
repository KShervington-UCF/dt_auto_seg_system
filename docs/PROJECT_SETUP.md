Project originally run and executed on Windows 10 ver. 10.0.19045 Build 19045 by following [tutorial from Tensorflow](https://www.tensorflow.org/install/pip).

- Python version 3.9.18 (within conda env)
- CUDA Toolkit 11.2
- cuDNN 8.1.0
- Tensorflow 2.10.1
- Miniconda3 (conda ver. 24.7.1)

# Project Setup (Windows)

1. Install Miniconda
   - Check miniconda installation by executing `conda info` in a command prompt. If this doesn't work, try launching a **Python Command Prompt** and executing `conda init` to initialize your command prompt with conda capabilities.
2. Launch a command prompt in this project's root folder
3. Create a local virtual environment: `conda create --prefix .\env python=3.9` (_replace 'env' with your preferred environment name if desired_)
4. Activate the virtual environment: `conda activate .\env`
5. (_Optional but recommended_) Enable GPU support: `conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0`
6. Install helpful packages: `conda install ipykernel pip`
   - `ipykernel`: for running kernels from conda
   - `pip`: for installing packages from `requirements.txt` and those that can't be installed with `conda`
7. Install project dependencies: `pip install -r requirements.txt`
8. (_Optional_) Verify GPU support:

   ```python
   import tensorflow as tf

   if len(tf.config.list_physical_devices('GPU')) > 0:
       print("GPU is Available!" )
   else:
       raise Exception("No GPU available")
   ```

# SAM2 Setup

Similar to above except for the following differences:

- Python version in step **3.** should `be 3.10`
- Dependencies to install in step 7 should be from the `sam2-requirements.txt` file
