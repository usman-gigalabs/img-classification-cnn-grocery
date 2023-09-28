import numpy as np
import os
import tensorflow as tf
from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical

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

        # Create an ImageDataGenerator for data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest'
        )

        # Iterate over image files in the leaf category
        for filename in image_files:
            img_path = os.path.join(directory, filename)

            # Load the image
            img = image.load_img(img_path, target_size=image_size)
            img_array = image.img_to_array(img)
            img_array /= 255.0  # Normalize pixel values to [0, 1]

            # Expand the dimensions to (1, height, width, channels)
            img_array = np.expand_dims(img_array, axis=0)

            # Apply data augmentation
            for batch in datagen.flow(img_array, batch_size=1):
                images.append(batch[0])
                # Create a label based on the current category index
                labels.append(index)
                break  # Exit the loop after applying augmentation

            # Append the image and its corresponding label to the lists
            # images.append(img_array)
            # labels.append(index)
        
        index += 1

# Start processing from the top-level data directory
process_subcategories(train_dir)

print(len(labels))
print(category_to_index)

# Convert lists to NumPy arrays
X_train = np.array(images)
y_train = np.array(labels)

print("Shape of X_train before squeezing:", X_train.shape)

# Check if squeezing is needed
if X_train.shape[1] == 1:
    X_train = np.squeeze(X_train, axis=1)

# Print the shape again to verify
print("Shape of X_train after squeezing:", X_train.shape)

# Calculate the number of classes (subcategories)
num_classes = len(category_to_index)
print(num_classes)

# One-hot encode the labels
y_train = to_categorical(y_train, num_classes=num_classes)

# Create a CNN model
model = Sequential()

# Add convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten the output
model.add(Flatten())

# Add fully connected layers
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))

# Output layer with num_classes neurons
model.add(Dense(num_classes, activation='softmax'))

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
model.save('my_image_classifier_model3.keras')

# Optional: Evaluate the model on test data
test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
)

test_steps_per_epoch = test_generator.samples // batch_size
test_loss, test_accuracy = model.evaluate(test_generator)
print(f'Test Loss: {test_loss:.4f}')
print(f'Test Accuracy: {test_accuracy:.4f}')


