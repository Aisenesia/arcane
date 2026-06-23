import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
import onnx


# Load the YOLO model
model = YOLO('runs/dice_detect.pt')
model.eval()
model.export(format='onnx', dynamic=True, opset=12)  # Export the model to ONNX format