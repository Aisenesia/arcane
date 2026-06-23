import os
import random
import shutil

# Define the paths
base_dir = "datasets_classify"
train_dir = os.path.join(base_dir, "train")
val_dir = os.path.join(base_dir, "val")

# Create the validation directory if it doesn't exist
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Loop through each class folder in the train directory
for class_name in os.listdir(train_dir):
    class_train_path = os.path.join(train_dir, class_name)
    class_val_path = os.path.join(val_dir, class_name)
    
    # Ensure it's a directory and not a file
    if not os.path.isdir(class_train_path):
        continue
    
    # Create corresponding class folder in validation directory
    if not os.path.exists(class_val_path):
        os.makedirs(class_val_path)
    
    # Get all image files in the class folder
    images = [img for img in os.listdir(class_train_path) if os.path.isfile(os.path.join(class_train_path, img))]
    
    # Randomly select 40 images
    selected_images = random.sample(images, 40)
    
    # Move the selected images to the validation folder
    for img in selected_images:
        src_path = os.path.join(class_train_path, img)
        dst_path = os.path.join(class_val_path, img)
        shutil.move(src_path, dst_path)

print("Dataset split completed!")