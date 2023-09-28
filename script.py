import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import numpy as np

import matplotlib.pyplot as plt

# Define the data directory and batch size
data_dir = "dataset"
validation_dir = "valset"
# test_dir = "GroceryStoreDataset/dataset/test"

batch_size = 32

# Define image dimensions
img_height, img_width = 348, 348

# Get a list of all subcategory names
subcategory_names = []
for root, dirs, files in os.walk(data_dir):
    if len(dirs) == 0:  # Assumes subcategories have no subdirectories
        subcategory_name = os.path.basename(root)
        subcategory_names.append(subcategory_name)

# Calculate the number of subcategories (classes)
# num_classes = len(subcategory_names)

print(subcategory_names)
# print(num_classes)

# Initialize an empty list to store generators for each main category
main_category_generators = []

# Iterate over the main categories
main_categories = os.listdir(data_dir)
for main_category in main_categories:
    main_category_dir = os.path.join(data_dir, main_category)
    
    # Create a data generator for the main category
    main_category_datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
    )

    main_category_generator = main_category_datagen.flow_from_directory(
        main_category_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',  # Use the training subset
    )
    
    main_category_generators.append(main_category_generator)

# Concatenate the generators
train_generator = main_category_generators[0]  # Start with the first generator
for generator in main_category_generators[1:]:
    train_generator = train_generator.concatenate(generator)

# Calculate the number of classes dynamically
num_classes = train_generator.num_classes

print(num_classes)

# validation_generator = train_datagen.flow_from_directory(
#     validation_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
#     subset='validation'  # Specify this is the validation set
# )

# for data_batch, labels_batch in train_generator:
#     print(f'Data batch shape: {data_batch.shape}')
#     print(f'Labels batch shape: {labels_batch.shape}')
#     break

# Build a Convolutional Neural Network (CNN)
# model = Sequential([
#     Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
#     # BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Conv2D(64, (3, 3), activation='relu'),
#     # BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Conv2D(128, (3, 3), activation='relu'),
#     # BatchNormalization(),
#     MaxPooling2D((2, 2)),
#     Flatten(),
#     Dense(512, activation='relu'),
#     # BatchNormalization(),
#     Dropout(0.5),
#     Dense(num_classes, activation='softmax') # 'num_classes' is the total number of categories and subcategories
# ])


# Compile the Model
# model.compile(
#     loss='categorical_crossentropy',
#     optimizer='adam',
#     metrics=['accuracy']
# )

# Train the Model
num_epochs = 10  # You can increase this for better accuracy

# Define the number of training and validation steps per epoch
# train_steps_per_epoch = train_generator.samples // batch_size
# val_steps_per_epoch = validation_generator.samples // batch_size

# history = model.fit(
#     train_generator,
#     epochs=num_epochs,
#     steps_per_epoch=train_steps_per_epoch,
#     validation_data=validation_generator,
#     validation_steps=val_steps_per_epoch,
# )

# Save the trained model
# model.save("model.keras")

# Evaluate the model on a test set if available
# test_generator = train_datagen.flow_from_directory(
#     test_dir,
#     target_size=(img_height, img_width),
#     batch_size=batch_size,
#     class_mode='categorical',
# )
# test_steps_per_epoch = test_generator.samples // batch_size
# test_loss, test_accuracy = model.evaluate(test_generator, steps=test_steps_per_epoch)
# print(f'Test Loss: {test_loss:.4f}')
# print(f'Test Accuracy: {test_accuracy:.4f}')

# Visualize training history
# acc = history.history['accuracy']
# val_acc = history.history['val_accuracy']
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# epochs = range(num_epochs)

# plt.figure(figsize=(12, 4))
# plt.subplot(1, 2, 1)
# plt.plot(epochs, acc, 'b', label='Training accuracy')
# plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
# plt.title('Training and Validation Accuracy')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.plot(epochs, loss, 'b', label='Training loss')
# plt.plot(epochs, val_loss, 'r', label='Validation loss')
# plt.title('Training and Validation Loss')
# plt.legend()

# plt.show()

