import torch
from ultralytics import YOLO

if __name__ == '__main__':
    # Load the trained YOLO model
    model = YOLO('best.pt')  # Replace with your trained model path

    # Train the model (if needed)
    # model.train(data="dataset.yaml", epochs=50, imgsz=640, batch=8) uncomment when training

    # Export the model to ONNX format
    model.export(format='onnx', imgsz=1280)  # Adjust imgsz if needed

    print("Model has been converted to ONNX format and saved.")