import os
from keras.preprocessing import image

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


# Now, you have images and labels ready for training
print(labels)
print(images)
print(category_to_index)
