import os
from PIL import Image

# Define the path to the dataset
dataset_path = 'datasets/train/images'

# Define the target dimensions
target_width = 3000
target_height = 2624

# Process each folder
for folder_name in os.listdir(dataset_path):
    folder_path = os.path.join(dataset_path, folder_name)
    if os.path.isdir(folder_path):
        # Process each image in the folder
        for filename in os.listdir(folder_path):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(folder_path, filename)
                with Image.open(image_path) as img:
                    # Crop the image
                    width, height = img.size
                    if width > target_width or height > target_height:
                        crop_area = (0, 0, target_width, target_height)
                        cropped_img = img.crop(crop_area)
                        # Save the cropped image with original file name appended with '_cropped'
                        name, ext = os.path.splitext(filename)
                        new_filename = f'{name}_cropped{ext}'
                        new_image_path = os.path.join(folder_path, new_filename)
                        cropped_img.save(new_image_path)
                        # Optionally, remove the original image
                        os.remove(image_path)