import cv2
import torch
from ultralytics import YOLO

# Load the YOLO model
model = YOLO('runs/classify/train2/weights/best.pt')  # Ensure the path is correct
model.eval()

# Open a connection to the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Perform inference (automatic preprocessing)
    with torch.no_grad():
        results = model(frame)  # Pass the raw frame

    # Access classification probabilities
    result = results[0]  # Extract first result
    if hasattr(result, 'probs') and result.probs is not None:
        probs = result.probs.data
        confidence, class_id = torch.max(probs, dim=0)  # Change dim=1 to dim=0
        confidence = confidence.item()
        class_id = class_id.item()

        if confidence > 0.5:  # Confidence threshold
            label = f'Class: {class_id}, Confidence: {confidence:.2f}'
            cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the result
    cv2.imshow('Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
