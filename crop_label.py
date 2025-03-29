import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('runs/dice_detect.pt')
model.eval()

is_crop = True  # Set to True to crop the images
is_save = True  # Set to True to save the cropped images

# Define the input and output directories
input_dir = 'datasets/train/images'

# Create the output directory if it doesn't exist

# Define additional output directories
cropped_dir = 'dataset_cropped'
labels_dir = 'datasets/train/labels'

# Create the additional directories if they don't exist
os.makedirs(cropped_dir, exist_ok=True)
os.makedirs(labels_dir, exist_ok=True)

# Function to preprocess and normalize the cropped image
def preprocess_and_normalize(cropped_img):
    resized_img = cv2.resize(cropped_img, (640, 640))
    return resized_img

# Function to save YOLO label files
def save_yolo_label(file_name, class_id, box, img_width, img_height):
    x_center = (box[0] + box[2]) / 2 / img_width
    y_center = (box[1] + box[3]) / 2 / img_height
    width = (box[2] - box[0]) / img_width
    height = (box[3] - box[1]) / img_height
    label_content = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
    
    # Create a directory for the class if it doesn't exist
    class_label_dir = os.path.join(labels_dir, str(class_id))
    os.makedirs(class_label_dir, exist_ok=True)


    
    # Save the label file in the class-specific directory
    label_file_path = os.path.join(class_label_dir, f"{file_name}.txt")
    with open(label_file_path, 'a') as label_file:
        label_file.write(label_content)

    

# Function to process images in a directory
def process_images_in_directory(directory):
    print(f"Input directory: {input_dir}")
    print(f"Files in input directory: {os.listdir(input_dir)}")
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.png'):
                file_path = os.path.join(root, file)
                img = cv2.imread(file_path)
                if img is None:
                    continue

                # Perform inference
                results = model(img)

                # Extract class ID from the folder name
                class_id = int(os.path.basename(root)) - 1

                # Process each detection
                for result in results:
                    for detection in result.boxes:
                        confidence = detection.conf
                        if confidence > 0.5:  # Confidence threshold
                            box = detection.xyxy[0].cpu().numpy().astype(int)

                            # Save YOLO label
                            save_yolo_label(os.path.splitext(file)[0], class_id, box, img.shape[1], img.shape[0])

                      
                            # Crop the image based on the square crop
                            x, y, x_max, y_max = box
                            crop_size = max(x_max - x, y_max - y)
                            center_x, center_y = (x + x_max) // 2, (y + y_max) // 2
                            x1 = max(center_x - crop_size // 2, 0)
                            y1 = max(center_y - crop_size // 2, 0)
                            x2 = min(center_x + crop_size // 2, img.shape[1])
                            y2 = min(center_y + crop_size // 2, img.shape[0])

                            cropped_img = img[y1:y2, x1:x2]

                            # Normalize the cropped image to 640x640
                            normalized_img = preprocess_and_normalize(cropped_img)

                            if is_save:
                                # Save the normalized image
                                class_dir = os.path.join(cropped_dir, str(class_id))
                                os.makedirs(class_dir, exist_ok=True)
                                output_file_path = os.path.join(class_dir, file)
                                cv2.imwrite(output_file_path, normalized_img)
                                print(f"Saved cropped image: {output_file_path}")



# Process images in the dataset2 directory
process_images_in_directory(input_dir)