import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('runs/detect/train7/weights/best.pt')
model.eval()

# Define the input and output directories
input_dir = 'dataset2'
output_dir = 'cropped_dataset2'

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Function to preprocess and normalize the cropped image
def preprocess_and_normalize(cropped_img):
    resized_img = cv2.resize(cropped_img, (640, 640))
    return resized_img

# Function to process images in a directory
def process_images_in_directory(directory):
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
                            x, y, x_max, y_max = box

                            # Calculate the size of the square crop
                            crop_size = max(x_max - x, y_max - y)
                            center_x, center_y = (x + x_max) // 2, (y + y_max) // 2

                            # Calculate the coordinates of the square crop
                            x1 = max(center_x - crop_size // 2, 0)
                            y1 = max(center_y - crop_size // 2, 0)
                            x2 = min(center_x + crop_size // 2, img.shape[1])
                            y2 = min(center_y + crop_size // 2, img.shape[0])

                            # Crop the image based on the square crop
                            cropped_img = img[y1:y2, x1:x2]

                            # Normalize the cropped image to 640x640
                            normalized_img = preprocess_and_normalize(cropped_img)

                            # Save the normalized image
                            class_dir = os.path.join(output_dir, str(class_id))
                            os.makedirs(class_dir, exist_ok=True)
                            output_file_path = os.path.join(class_dir, file)
                            cv2.imwrite(output_file_path, normalized_img)

# Process images in the dataset2 directory
process_images_in_directory(input_dir)