import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
import json
import sys
from pathlib import Path
from model_loader import ModelLoader
import datetime

# Add the project root to the Python path to import the config utils
project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

# Import the config utility
from modules.utils.config_utils import config

# Initialize model loader and load model
model_loader = ModelLoader()
loaded_model = model_loader.load_model()

def classify_image(image_path):
    # Generate single prediction based on image
    img = Image.open(image_path)
    img = img.resize((224, 224)) # ResNet input size
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create a batch

    predictions = loaded_model.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    return predictions, score

if __name__ == '__main__':
    class_labels = ["dry-asphalt-bad",
        "dry-asphalt-good",
        "dry-asphalt-intermediate",
        "dry-paved-bad",
        "dry-paved-good",
        "dry-paved-intermediate",
        "unpaved",
        "water-asphalt",
        "water-paved",
        "water-unpaved",
        "wet-asphalt-bad",
        "wet-asphalt-good",
        "wet-asphalt-intermediate",
        "wet-paved-bad",
        "wet-paved-good",
        "wet-paved-intermediate"]

    # Get input directory from config
    test_images_dir = config.get_path('classification', 'input_dir')
    
    total_images = 0

    json_output_array = []

    for image_file in os.listdir(test_images_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_images_dir, image_file)
            
            predictions, scores = classify_image(image_path)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            confidence_score = scores[predicted_class_index]

            total_images += 1

            json_output_array.append({
                "image": image_file,
                "predicted_class": predicted_class_label,
                "confidence_score": round(confidence_score.numpy().item(), 4)
            })

    # Create output directory if it doesn't exist
    output_dir = config.get_path('classification', 'output_dir')
    os.makedirs(output_dir, exist_ok=True)

    # Create a timestamp for the prediction file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_dir, f"image_classifications_{timestamp}.json")
    
    # Also save to the standard path defined in config
    standard_path = config.get_path('classification', 'classification_file_path')
    os.makedirs(os.path.dirname(standard_path), exist_ok=True)

    # Save to both locations
    with open(output_file_path, 'w') as output_file:
        json.dump(json_output_array, output_file, indent=2)
        
    with open(standard_path, 'w') as std_file:
        json.dump(json_output_array, std_file, indent=2)
    
    print(f"Classified {total_images} images and saved predictions to:\n- {output_file_path}\n- {standard_path}")
