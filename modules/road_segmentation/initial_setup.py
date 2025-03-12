import urllib.request
import os

# Function to download a file from a url
def download_file(url, filename):
    """
    Downloads a file from a URL and saves it to the specified filename.

    Args:
        url (str): The URL of the file to download.
        filename (str): The name of the file to save the downloaded content as.
    """
    try:
        urllib.request.urlretrieve(url, filename)
        print(f"File downloaded successfully to {filename}")
    except Exception as e:
        print(f"Error downloading file: {e}")

if __name__ == '__main__':
    # Create model checkpoint directory
    model_checkpoint_dir = os.path.join(os.path.dirname(__file__), 'model')
    os.makedirs(model_checkpoint_dir, exist_ok=True)

    # Check if model file exists before downloading it
    model_file_path = os.path.join(model_checkpoint_dir, 'sam2.1_hiera_large.pt')
    if not os.path.exists(model_file_path):
        download_file('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt', model_file_path)

    print("Road segmentation module initialized")