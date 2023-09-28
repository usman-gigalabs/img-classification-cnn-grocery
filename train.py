import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.utils import to_categorical
from tensorflow.keras.models import load_model

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

# Iterate over the main categories
main_categories = os.listdir(train_dir)
for main_category in main_categories:
    main_category_dir = os.path.join(train_dir, main_category)
    
    # Check if there are subcategories within the main category
    subcategories = os.listdir(main_category_dir)
    
    if len(subcategories) == 0:
        # No subcategories, assign a label to the main category itself
        category_to_index[main_category] = index
        
        # Iterate over images in the main category
        for filename in os.listdir(main_category_dir):
            img_path = os.path.join(main_category_dir, filename)
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            img_array /= 255.0  # Normalize pixel values to [0, 1]
            images.append(img_array)
            
            # Create a label based on the main category index
            label = index
            labels.append(label)
        
        index += 1
    else:
        # Subcategories exist, assign labels to subcategories
        for subcategory in subcategories:
            subcategory_dir = os.path.join(main_category_dir, subcategory)
            
            # Store the subcategory name and its corresponding index
            category_to_index[subcategory] = index
            
            # Iterate over images in the subcategory
            for filename in os.listdir(subcategory_dir):
                img_path = os.path.join(subcategory_dir, filename)
                img = image.load_img(img_path, target_size=image_size)
                img_array = image.img_to_array(img)
                img_array /= 255.0  # Normalize pixel values to [0, 1]
                images.append(img_array)
                
                # Create a label based on the subcategory index
                label = index
                labels.append(label)
            
            index += 1

# Convert lists to NumPy arrays
X_train = np.array(images)
y_train = np.array(labels)

# Calculate the number of classes (subcategories)
num_classes = len(category_to_index)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=num_classes)

# Load the saved model
model = load_model("my_image_classifier_model.keras")

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=10,  # You can increase the number of epochs as needed
)

# Save the trained model
model.save('my_image_classifier_model.keras')

# Optional: Evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
)

test_steps_per_epoch = test_generator.samples // batch_size
test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps_per_epoch)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')


