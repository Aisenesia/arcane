import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained YOLO model
    model = YOLO('yolo11m.pt')  # Replace with your trained model path

    # Train the model (if needed)
    model.train(data="dataset.yaml", epochs=50, imgsz=640, batch=8, single_cls=True) # uncomment when training

    results = model.val()
    print(results)