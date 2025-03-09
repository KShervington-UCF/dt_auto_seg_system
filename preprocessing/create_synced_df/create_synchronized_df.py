# A script which uses raw Sensor Logger data to create a synchronized dataframe

# Import relevant packages
import pandas as pd
import os
import json
import datetime

# Get the directory of the script
curr_directory = os.path.dirname(__file__)

# 1. Load configuration file
config_file = os.path.join(curr_directory, 'config.json')
with open(config_file, 'r') as f:
    config = json.load(f)

# 2. Import raw data files
location_file = os.path.join(curr_directory, config['location_file_path'])
orientation_file = os.path.join(curr_directory, config['orientation_file_path'])
image_folder = os.path.join(curr_directory, config['image_folder_path'])

# 3. 
image_names = os.listdir(image_folder)

# Load csv data into dataframes
location_df = pd.read_csv(location_file)
orientation_df = pd.read_csv(orientation_file)

# Function to convert UTC time to GPS time
def utc_to_gps(utc_time):
    """
    :param utc_time: A datetime object representing UTC time.
    :return: A datetime object representing GPS time.
    """

    # GPS epoch (January 6, 1980)
    GPS_EPOCH = datetime.datetime(1980, 1, 6, 0, 0, 0)

    # Current leap second difference between GPS and UTC (there is a 18-second difference)
    LEAP_SECONDS = 18

    # Calculate the difference between UTC and GPS epoch
    delta = utc_time - GPS_EPOCH

    # Subtract leap seconds to get GPS time
    gps_time = delta.total_seconds() + LEAP_SECONDS

    return int(gps_time)

# Function which converts epoch time (in nanoseconds) into gps time
def epoch_ns_to_gps_time(epoch_ns):
    """
    Convert epoch time in nanoseconds to GPS time.
    
    Args:
        epoch_ns (int): Epoch time in nanoseconds (nanoseconds since January 1, 1970)
        
    Returns:
        tuple: (gps_week, gps_seconds) where:
            - gps_week (int): GPS week number
            - gps_seconds (float): Seconds into the GPS week
    """
    epoch = str(epoch_ns)
    epoch = epoch[:10]
    epoch = int(epoch)
    date, time = datetime.datetime.fromtimestamp(epoch).strftime('%Y-%m-%d'), datetime.datetime.fromtimestamp(epoch).strftime('%H:%M:%S')

    # Combine the date and time into a datetime object
    utc_datetime = datetime.datetime.strptime(f"{date} {time}", '%Y-%m-%d %H:%M:%S')

    # Convert to GPS time using the utc_to_gps function
    gps_time = utc_to_gps(utc_datetime)

    return gps_time

# Convert epoch time to gps time in both dataframes
location_df['gps_time'] = location_df['time'].apply(epoch_ns_to_gps_time)
orientation_df['gps_time'] = orientation_df['time'].apply(epoch_ns_to_gps_time)

# Merge location and orientation dataframes
loc_ori_merged_df = pd.merge(location_df, orientation_df, on='gps_time', how='inner')

print(loc_ori_merged_df.head())

# Drop all columns except those defined in config["column_names"]
# loc_ori_merged_df = loc_ori_merged_df[config['column_names']]

# 4. Create synchronized dataframe

# 5. Save synchronized dataframe to CSV
