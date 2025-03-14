import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('best.pt')
model.eval()

# Define a function to preprocess the frame
def preprocess_frame(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (640, 640))  # Adjust size if needed
    frame = frame / 255.0  # Normalize to [0, 1]
    frame = np.transpose(frame, (2, 0, 1))  # Change to (C, H, W)
    frame = np.expand_dims(frame, axis=0)  # Add batch dimension
    frame = torch.tensor(frame, dtype=torch.float32)
    return frame

# Open a connection to the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess the frame
    input_tensor = preprocess_frame(frame)

    # Perform inference
    with torch.no_grad():
        results = model(input_tensor)

    # Post-process the output
    for result in results:
        for detection in result.boxes:  # Assuming result.boxes contains the detections
            confidence = detection.conf
            if confidence > 0.5:  # Confidence threshold
                class_id = int(detection.cls)
                box = detection.xyxy[0].cpu().numpy().astype(int)
                (x, y, x_max, y_max) = box

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)
                label = f'Dice Face: {class_id + 1}'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Dice Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()