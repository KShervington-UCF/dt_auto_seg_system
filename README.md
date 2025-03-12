python version 3.9.18

Sam2 environment requires python version >=3.10

[Project Setup](./docs/PROJECT_SETUP.md)

Process Flow for manual testing:

1. Place Sensor Logger data in the `preprocessing\create_synced_df\sample_raw_data` folder
   - The data should include at least the following files/directories: `Camera/`, `Location.csv`, `Orientation.csv`
2. Execute `preprocessing\create_synced_df\create_synchronized_df.py` script
3. Copy `preprocessing\create_synced_df\output\synchronized_df.csv` to `modules\geopose`
4. Execute `modules\geopose\create_geopose.py` to create a geopose file
5. Execute `preprocessing\crop_images\crop_images.py` script to crop images from `preprocessing\create_synced_df\sample_raw_data\Camera`
6. All or a subset of images can be copied from `preprocessing\crop_images\output` to `modules\classification\test_images`
7. Execute `modules\classification\classify_images.py` to create a classification report
