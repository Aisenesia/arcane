import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained YOLO model
    model = YOLO('yolo11s-cls.pt')  # Replace with your trained model path

    # Train the model (if needed)
    model.train(data="datasets_classify/", epochs=50, imgsz=224, batch=16)


    results = model.val()
    print(results)