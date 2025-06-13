import pandas as pd
import numpy as np
import os
import hashlib
import json
import sys
from pathlib import Path

# Add the project root to the Python path to import the config utils
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the config utility
from modules.utils.config_utils import config

# Load synchronized dataframe file created from preprocessing
synced_df_path = config.get_path('preprocessing', 'synced_df_path')
synced_df = pd.read_csv(synced_df_path)

# Define the reference point (first entry in the dataset)
lat_ref = synced_df.iloc[0]['latitude']
lon_ref = synced_df.iloc[0]['longitude']
alt_ref = synced_df.iloc[0]['altitude']

# Calculate inter-pose duration (assuming regular interval based on GPS time difference)
# NOTE: total_samples = ((last_timestamp - initial_timestamp)/consecutive_timestamp_diff) + 1
timestamps = synced_df['gps_time'].values
inter_pose_duration = int(np.median(np.diff(timestamps)))  # Use median to find the regular interval

# Extract the start and stop instants
start_instant = int(str(timestamps[0])[:10])
stop_instant = int(str(timestamps[-1])[:10])

# Count the number of poses
pose_count = len(timestamps)

# Generate a SHA256 integrity check based on the pose data
pose_data_str = "".join([f"{row['latitude']}{row['longitude']}{row['altitude']}"
                          f"{row['qx']}{row['qy']}{row['qz']}{row['qw']}" for index, row in synced_df.iterrows()])

integrity_check = hashlib.sha256(pose_data_str.encode()).hexdigest()

# Define the outer frame using the reference LTP (Latitude, Longitude, Altitude)
outer_frame = {
    "authority": "/geopose/1.0",
    "id": "LTP-ENU",
    "parameters": f"longitude={lon_ref}&latitude={lat_ref}&height={alt_ref}"
}

# Define the inner frame series (translation and rotation for each pose)
inner_frame_series = []
for i, row in synced_df.iterrows():
    # E, N, U = enu_coordinates[i]
    E, N, U = row['latitude'], row['longitude'], row['altitude']
    inner_frame_series.append({
        "authority": "/geopose/1.0",
        "id": "RotateTranslate",
        "parameters": f"translation=[{E}, {N}, {U}]&rotation=[{row['qx']}, {row['qy']}, {row['qz']}, {row['qw']}]"
      })

# Create the Series Header
series_header = {
    "poseCount": pose_count,
    "integrityCheck": f"{{\"SHA256\": \"{integrity_check}\"}}",
    "startInstant": start_instant,
    "stopInstant": stop_instant,
    "transitionModel": {
        "authority": "/geopose/1.0",
        "id": "none",
        "parameters": ""
    }
}

# Create the Series Trailer
series_trailer = {
    "poseCount": pose_count,
    "integrityCheck": f"{{\"SHA256\": \"{integrity_check}\"}}"
}

# Assemble the GeoPose JSON structure
geopose_series = {
    "header": series_header,
    "interPoseDuration": inter_pose_duration,
    "outerFrame": outer_frame,
    "innerFrameSeries": inner_frame_series,
    "trailer": series_trailer
}

# Create output directory if it doesn't exist
output_dir = config.get_path('geopose', 'output_dir')
os.makedirs(output_dir, exist_ok=True)

# Save geopose file to configured path
geopose_path = config.get_path('geopose', 'geopose_file_path')
os.makedirs(os.path.dirname(geopose_path), exist_ok=True)
with open(geopose_path, 'w') as json_file:
    json.dump(geopose_series, json_file, indent=4)

print(f'Geopose file saved to {geopose_path}')

# Original script author: Kalp Devangbhai Thakkar 
