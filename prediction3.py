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
loaded_model = load_model('my_image_classifier_model2old.keras')

# Define data directories
train_dir = 'dataset'
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
def process_subcategories(directory, parent_category_index=None):
    global index

    # Get the list of subdirectories and files in the current directory
    items = os.listdir(directory)

    # Check if there are subdirectories (subcategories)
    subdirectories = [item for item in items if os.path.isdir(os.path.join(directory, item))]

    if subdirectories:
        # If there are subdirectories, process each subcategory
        for subcategory in subdirectories:
            subcategory_dir = os.path.join(directory, subcategory)
            
            # Store the subcategory name and its corresponding index
            # category_to_index[subcategory] = index
            
            # Recursively process the subcategory
            process_subcategories(subcategory_dir, index)
            
            index += 1
    else:
        # No subdirectories, assign a label to the current category
        category_to_index[os.path.basename(directory)] = index

        # Iterate over images in the current category
        for filename in items:
            img_path = os.path.join(directory, filename)
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            img_array /= 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)

            # Create a label based on the current category index
            label = parent_category_index if parent_category_index is not None else index
            labels.append(label)

# Start processing from the top-level data directory
process_subcategories(train_dir)

# Load and preprocess the input image for prediction
input_image_path = 'GroceryStoreDataset/sample_images/natural/Green-Bell-Pepper.jpg'
# input_image_path = 'GroceryStoreDataset/dataset/test/Fruit/Pear/Conference/Conference_014.jpg'
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