import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load your trained model
model = load_model('model.keras')

# Define the data directory and batch size
data_dir = "GroceryStoreDataset/dataset/train"

# Define image dimensions
img_height, img_width = 348, 348

# Load and preprocess a test image
# img_path = 'PepsiandCocaColaImages/test/pepsi/20.jpg'
# img_path = '/Users/gigalabs/Downloads/pepsi-coke.jpeg'
img_path = 'GroceryStoreDataset/sample_images/iconic/Alpro-Fresh-Soy-Milk_Iconic.jpg'
img = image.load_img(img_path, target_size=(img_height, img_width))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values to [0, 1]

# Make predictions
predictions = model.predict(img_array)
print(predictions)

# Get the predicted class index
predicted_class_index = np.argmax(predictions)

# Get a list of all subcategory names
subcategory_names = []
for root, dirs, files in os.walk(data_dir):
    if len(dirs) == 0:  # Assumes subcategories have no subdirectories
        subcategory_name = os.path.basename(root)
        subcategory_names.append(subcategory_name)

# Predicted subcategory name
predicted_subcategory_name = subcategory_names[predicted_class_index]

# Display the predicted subcategory name
print(f'Predicted class: {predicted_subcategory_name}')
