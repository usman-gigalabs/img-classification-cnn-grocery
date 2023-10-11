import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np


# Use the trained model for prediction
from keras.models import load_model

image_size = (348, 348)

# Load the saved model
loaded_model = load_model('my_image_classifier_model2.keras')

# Define data directories
train_dir = 'GroceryStoreDataset/dataset/train'
test_dir = 'GroceryStoreDataset/sample_images/natural'  # Optional, for testing the saved model

# Define image size and batch size
image_size = (348, 348)
batch_size = 32

# Initialize lists to store images and labels
images = []
labels = []

# Create a mapping from category name to its index
category_to_index = {}
index = 0

# Define a recursive function to process subcategories
def process_subcategories(directory, parent_category_index=None, parent_category_name=None):
    global index

    # Get the list of subdirectories and files in the current directory
    items = os.listdir(directory)

    # Filter out only image files (you can add more extensions if needed)
    image_files = [item for item in items if item.lower().endswith(('.jpg', '.jpeg', '.png', '.gif'))]

    # Check if there are subdirectories (subcategories)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory, item))]

    if subdirectories:
        # If there are subdirectories, process each subcategory
        for subcategory in subdirectories:
            subcategory_dir = os.path.join(directory, subcategory)

            subcategory_name = os.path.join(parent_category_name, subcategory) if parent_category_name else subcategory

            # Recursively process the subcategory
            process_subcategories(subcategory_dir, index, subcategory_name)

    else:
        # If no subdirectories, this is a leaf category
        

        # Assign a label to the current leaf category
        # category_name = os.path.basename(directory)
        # category_to_index[category_name] = index

        category_to_index[parent_category_name] = index if parent_category_name else index

        # Iterate over image files in the leaf category
        for filename in image_files:
            img_path = os.path.join(directory, filename)

            # Load the image
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            img_array /= 255.0  # Normalize pixel values to [0, 1]

            # Expand the dimensions to (1, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)

            # Append the image and its corresponding label to the lists
            images.append(img_array)
            labels.append(index)
        
        index += 1


# Start processing from the top-level data directory
process_subcategories(train_dir)

# Load and preprocess the input image for prediction
# input_image_path = '/Users/gigalabs/Downloads/download.jpeg'
# input_image_path = 'GroceryStoreDataset/sample_images/natural/Yellow-Onion.jpg'
input_image_path = 'GroceryStoreDataset/dataset/test/Vegetables/Ginger/Ginger_006.jpg'
img = image.load_img(input_image_path, target_size=image_size)
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Make predictions using the loaded model
predictions = loaded_model.predict(img_array)
print(predictions)

# Get the predicted subcategory class index
predicted_class_index = np.argmax(predictions)

# Reverse the mapping to get the category name for the predicted class index
predicted_category = [category for category, index in category_to_index.items() if index == predicted_class_index]

if predicted_category:
    print(f"The predicted category is: {predicted_category[0]}")
else:
    print("Category not found for the predicted class index.")