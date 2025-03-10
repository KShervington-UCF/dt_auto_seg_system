import tensorflow as tf
from tensorflow import keras
from PIL import Image
import numpy as np
import os
from model_loader import ModelLoader

# Get the directory of the script
script_dir = os.path.dirname(os.path.abspath(__file__))

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

def get_ground_truth(filename):
    # Extract ground truth label from filename
    # Remove the timestamp and .jpg extension
    return filename.split('-', 1)[1].rsplit('.', 1)[0]

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

    # Some images pulled from the training dataset
    test_images_dir = os.path.join(script_dir, 'test_images')
    
    # Process all images in the test_images directory
    correct_predictions = 0
    total_images = 0
    
    print("\nClassification Results:")
    print("-" * 80)
    print(f"{'Image File':<40} {'Predicted':<20} {'Ground Truth':<20}")
    print("-" * 80)

    for image_file in os.listdir(test_images_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(test_images_dir, image_file)
            ground_truth = get_ground_truth(image_file)
            
            predictions, scores = classify_image(image_path)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            confidence_score = scores[predicted_class_index]

            is_correct = predicted_class_label == ground_truth
            correct_predictions += int(is_correct)
            total_images += 1

            # Print results with formatting
            print(f"{image_file:<40} {predicted_class_label:<20} {ground_truth:<20}")
            print(f"Confidence Score: {confidence_score:.2%}")
            print("-" * 80)

    # Print summary
    accuracy = correct_predictions / total_images if total_images > 0 else 0
    print(f"\nSummary:")
    print(f"Total images processed: {total_images}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Accuracy: {accuracy:.2%}")
