import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained YOLO model
    model = YOLO('yolo11s-cls.pt')  # Replace with your trained model path

    # Train the model (if needed)
    model.train(data="datasets/", epochs=50, imgsz=640, batch=8)


    results = model.val()
    print(results)