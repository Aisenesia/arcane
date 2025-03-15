import os
import torch
from ultralytics import YOLO

# Load the trained model
model = YOLO('runs/classify/train2/weights/best.pt')  # Adjust the path

# Path to the test image
test_image_path = 'datasets/val/16/img12.jpg'

# Verify the image path
if not os.path.exists(test_image_path):
    print(f"Error: Image {test_image_path} does not exist.")
else:
    print(f"Processing image: {test_image_path}")

    # Perform inference
    results = model(test_image_path)

    # Debugging output
    print(f"Raw Results: {results}")

    if results and len(results) > 0:
        predicted_class = results[0].probs.top1  # Best predicted class
        confidence = results[0].probs.top1conf  # Confidence level
        class_names = model.names  # Class labels
        predicted_label = class_names[predicted_class] if class_names else predicted_class
        print(f"Predicted Class: {predicted_label}, Confidence: {confidence:.2f}")
    else:
        print(f"No classification for image: {test_image_path}")
