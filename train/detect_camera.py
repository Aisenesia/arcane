import cv2
import numpy as np
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('runs/detect/train7/weights/best.pt')
model.eval()

# Open a connection to the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Get original dimensions
    original_h, original_w, _ = frame.shape

    # Perform inference without manual resizing
    results = model(frame)  # YOLO handles resizing internally

    # Post-process the output
    for result in results:
        for detection in result.boxes:
            confidence = detection.conf
            if confidence > 0.5:  # Confidence threshold
                class_id = int(detection.cls)
                box = detection.xyxy[0].cpu().numpy()

                # Scale bounding box coordinates back to original size
                x, y, x_max, y_max = map(int, box)

                # Draw the bounding box and label on the frame
                cv2.rectangle(frame, (x, y), (x_max, y_max), (0, 255, 0), 2)
                label = f'Dice Face: {class_id + 1}'
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw a plus icon at the center of the frame
    center_x, center_y = original_w // 2, original_h // 2
    cv2.line(frame, (center_x - 10, center_y), (center_x + 10, center_y), (0, 0, 255), 2)
    cv2.line(frame, (center_x, center_y - 10), (center_x, center_y + 10), (0, 0, 255), 2)

    # Display the result
    cv2.imshow('Dice Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
