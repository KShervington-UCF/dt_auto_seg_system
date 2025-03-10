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

# 3. Perform conversions and merges

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
def epoch_ns_to_utc(epoch_ns):
    epoch = str(epoch_ns)
    epoch = epoch[:10]
    epoch = int(epoch)
    return datetime.datetime.fromtimestamp(epoch).strftime('%Y-%m-%d'), datetime.datetime.fromtimestamp(epoch).strftime('%H:%M:%S')

def utc_to_gps_time(row):
    # Combine the date and time into a datetime object
    utc_datetime = datetime.datetime.strptime(f"{row['date']} {row['time']}", '%Y-%m-%d %H:%M:%S')

    # Convert to GPS time using the utc_to_gps function
    gps_time = utc_to_gps(utc_datetime)

    return gps_time

# Rename time column to timestamp
location_df.rename(columns={'time': 'timestamp'}, inplace=True)
orientation_df.rename(columns={'time': 'timestamp'}, inplace=True)

# Create date and time columns
location_df[['date', 'time']] = location_df['timestamp'].apply(lambda x: pd.Series(epoch_ns_to_utc(x)))
orientation_df[['date', 'time']] = orientation_df['timestamp'].apply(lambda x: pd.Series(epoch_ns_to_utc(x)))

# Merge location and orientation dataframes
loc_ori_merged_df = pd.merge(location_df, orientation_df, on=['date', 'time'], how='inner')
loc_ori_merged_df = loc_ori_merged_df.dropna()

loc_ori_merged_df['gps_time'] = loc_ori_merged_df.apply(utc_to_gps_time, axis=1)

# Move 'GPS Time' to the first column
cols = loc_ori_merged_df.columns.tolist()

# Remove 'GPS Time' from its original position and insert it at the first position
cols.insert(0, cols.pop(cols.index('gps_time')))

# Reorder the DataFrame using the updated column order
loc_ori_merged_df = loc_ori_merged_df[cols]

# Get list of image names
image_names = os.listdir(image_folder)

# Create a dictionary to map GPS times to image names
image_gps_times = {}
for image_name in image_names:
    if image_name.endswith('.jpg'):
        # Extract epoch time from image name (remove .jpg extension)
        epoch_time_str = image_name.split('.')[0]
        try:
            # Convert to integer and then to GPS time
            epoch_time = int(epoch_time_str)
            # Convert milliseconds to nanoseconds for the epoch_ns_to_gps_time function
            epoch_ns = epoch_time * 1000000  # Convert milliseconds to nanoseconds
            img_date, img_time = epoch_ns_to_utc(epoch_ns)

            sim_row = {
                'date': img_date,
                'time': img_time,
            }

            image_gps_time = utc_to_gps_time(sim_row)
            image_gps_times[image_gps_time] = image_name
        except ValueError:
            print(f"Could not convert image name {image_name} to GPS time")

# Function to find the closest GPS time match for each row
def find_closest_image(row_gps_time, image_gps_times_dict):
    if not image_gps_times_dict:
        return None
    
    # Find the closest GPS time
    closest_gps_time = min(image_gps_times_dict.keys(), key=lambda x: abs(x - row_gps_time))
    
    # Only return a match if it's within a reasonable time difference (e.g., 1 second)
    # Adjust this threshold as needed based on your data collection frequency
    if abs(closest_gps_time - row_gps_time) <= 1:  # 1 second threshold
        return image_gps_times_dict[closest_gps_time]
    else:
        return None

# 4. Create synchronized dataframe
# Add the image column to the location-orientation dataframe
loc_ori_merged_df['image'] = loc_ori_merged_df['gps_time'].apply(
    lambda x: find_closest_image(x, image_gps_times)
)

fully_merged_df = loc_ori_merged_df.copy()

# Drop all columns except those defined in config["column_names"]
fully_merged_df = fully_merged_df[config['column_names']]

# Remove duplicates to ensure images only appear once
fully_merged_df = fully_merged_df.drop_duplicates(subset=['image'])

# 5. Save synchronized dataframe to CSV
synced_df_output_path = os.path.join(curr_directory, config['syncronized_df_parentDir'], 'synchronized_df.csv')
fully_merged_df.to_csv(synced_df_output_path, index=False)

print(f"Synchronized dataframe saved to {synced_df_output_path}")

