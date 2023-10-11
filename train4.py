import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.utils import to_categorical
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Define data directories
train_dir = 'GroceryStoreDataset/dataset/train'
test_dir = 'GroceryStoreDataset/dataset/test'  # Optional, for testing the saved model

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

print(category_to_index)

# Convert lists to NumPy arrays
X_train = np.array(images)
y_train = np.array(labels)

X_train = np.squeeze(X_train, axis=1)

# Calculate the number of classes (subcategories)
num_classes = len(category_to_index)
print(num_classes)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=num_classes)

# Load the saved model
model = load_model("my_image_classifier_model2.keras")

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
num_epochs = 10

# Train the model
history = model.fit(
    X_train,
    y_train,
    batch_size=batch_size,
    epochs=num_epochs,  # You can increase the number of epochs as needed
)

# Save the trained model
model.save('my_image_classifier_model2.keras')

# Optional: Evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

images = []
labels = []
index = 0
category_to_index = {}
process_subcategories(test_dir)

# Convert lists to NumPy arrays
x_test = np.array(images)
y_test = np.array(labels)

x_test = np.squeeze(x_test, axis=1)

# Calculate the number of classes (subcategories)
num_classes = len(category_to_index)
print(num_classes)

# One-hot encode the labels
y_test = to_categorical(y_test, num_classes=num_classes)

test_generator = test_datagen.flow(
    x_test,  # Your synchronized test data
    y_test,  # Your synchronized test labels (should be one-hot encoded)
    batch_size=batch_size,
    shuffle=False
)

# test_steps_per_epoch = test_generator.samples // batch_size
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')

# Visualize training history
acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
loss = history.history['loss']
# val_loss = history.history['val_loss']

epochs = range(num_epochs)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'b', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()

plt.show()


