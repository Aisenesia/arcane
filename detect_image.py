import os
import torch
from ultralytics import YOLO

# Detects objects in a test image using a trained YOLO model instead of a camera feed


# Load the trained model
model = YOLO('runs/classify/train17/weights/best.pt')  # Adjust the path to your trained model

# Path to the test image
test_image_path = 'datasets/val/13/img1.jpg'  # Adjust the path to your test image

# Verify the image path
if not os.path.exists(test_image_path):
    print(f"Error: Image {test_image_path} does not exist.")
else:
    print(f"Processing image: {test_image_path}")

    # Perform inference
    results = model(test_image_path)
    print(f"Results: {results}")

    # Assuming the model returns the class with the highest confidence
    if len(results[0].boxes.data) > 0:
        # Print the structure of the results to debug
        print(f"Boxes data: {results[0].boxes.data}")
        predicted_class = int(results[0].boxes.data[0, 5].item())
        print(f"Predicted class: {predicted_class}")
    else:
        print(f"No detection for image: {test_image_path}")