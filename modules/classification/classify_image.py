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

if __name__ == '__main__':
    image_path = os.path.join(script_dir, '..', '..', 'test_images', '202201271541177-dry-asphalt-bad.jpg')
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
    predictions, scores = classify_image(image_path)

    predicted_class_index = np.argmax(predictions)
    predicted_class_label = class_labels[predicted_class_index]

    confidence_score = scores[predicted_class_index]  # Get score for predicted class

    print(f'Predicted class: {predicted_class_label}')
    print(f'Score: {confidence_score}')
