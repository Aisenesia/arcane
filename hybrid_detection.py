import cv2
import torch
from ultralytics import YOLO

# Load the YOLO models
detect_model = YOLO('runs/dice_detect.pt')  # Detection model
classify_model = YOLO('runs/dice_classify_2.pt')  # Classification model
detect_model.eval()
classify_model.eval()

class_mapping = {
    0: 1, 1: 2, 12: 3, 13: 4, 14: 5, 15: 6,
    16: 7, 17: 8, 18: 9, 19: 10,
    2: 11, 3: 12, 4: 13, 5: 14, 6: 15,
    7: 16, 8: 17, 9: 18, 10: 19, 11: 20
}

# Open a connection to the camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Perform detection (automatic preprocessing)
    with torch.no_grad():
        detect_results = detect_model(frame)

    # Post-process detection results
    for detect_result in detect_results:  # Iterate over detection results
        if hasattr(detect_result, 'boxes') and detect_result.boxes is not None:
            for box in detect_result.boxes:  # Iterate over detected boxes
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                cropped_frame = frame[y1:y2, x1:x2]  # Crop the detected region

                # Perform classification if the cropped frame is valid
                if cropped_frame.size > 0:
                    with torch.no_grad():
                        classify_results = classify_model(cropped_frame)  # Pass raw cropped frame

                    # Access classification probabilities
                    result = classify_results[0]  # Extract first result
                    if hasattr(result, 'probs') and result.probs is not None:
                        probs = result.probs.data
                        confidence, class_id = torch.max(probs, dim=0)
                        confidence = confidence.item()
                        class_id = class_id.item()

                        if confidence > 0.5:  # Confidence threshold
                            mapped_class = class_mapping.get(class_id, class_id)  # Default to class_id if not in mapping
                            label = f'Class: {mapped_class}, Confidence: {confidence:.2f}'
                            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                # Draw the bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Display the result
    cv2.imshow('Detection and Classification', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
