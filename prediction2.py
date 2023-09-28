import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np


# Use the trained model for prediction
from keras.models import load_model

image_size = (348, 348)

# Load the saved model
loaded_model = load_model('my_image_classifier_model.keras')

# Define data directories
train_dir = 'dataset'

# Define image size and batch size
image_size = (348, 348)
batch_size = 32

# Initialize lists to store images and labels
images = []
labels = []

# Create a mapping from subcategory name to its index
subcategory_to_index = {}
index = 0

# Iterate over the main categories
main_categories = os.listdir(train_dir)
for main_category in main_categories:
    main_category_dir = os.path.join(train_dir, main_category)
    
    # Iterate over subcategories
    subcategories = os.listdir(main_category_dir)
    for subcategory in subcategories:
        subcategory_dir = os.path.join(main_category_dir, subcategory)
        
        # Store the subcategory name and its corresponding index
        subcategory_to_index[subcategory] = index
        
        # Iterate over images in the subcategory
        for filename in os.listdir(subcategory_dir):
            img_path = os.path.join(subcategory_dir, filename)
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            img_array /= 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)
            
            # Create a one-hot encoded label based on the subcategory
            label = index
            labels.append(label)
        
        index += 1

# Load and preprocess the input image for prediction
input_image_path = 'GroceryStoreDataset/sample_images/natural/Banana.jpg'
img = image.load_img(input_image_path, target_size=image_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions using the loaded model
predictions = loaded_model.predict(img_array)

# Get the predicted subcategory class index
predicted_class_index = np.argmax(predictions)

# Map the class index back to the subcategory name
predicted_subcategory_name = list(subcategory_to_index.keys())[list(subcategory_to_index.values()).index(predicted_class_index)]

# Display the predicted subcategory name
print(f'Predicted subcategory: {predicted_subcategory_name}')