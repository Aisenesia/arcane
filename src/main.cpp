#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Load and run classification model
void classifyImage(const string& modelPath, const Mat& image) {
    Net net = readNetFromONNX(modelPath);
    net.setPreferableBackend(DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);

    Mat blob;
    blobFromImage(image, blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
    net.setInput(blob);
    
    Mat output = net.forward();

    // Get the predicted class
    Point classIdPoint;
    double confidence;
    minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int classId = classIdPoint.x;

    cout << "Classified as: " << classId << " with confidence: " << confidence << endl;

    // Display classification result
    putText(image, "Class: " + to_string(classId), Point(10, 30),
            FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
    imshow("Classified Dice", image);
}

// Load and run detection model
void detectDice(const string& detectionModelPath, const string& classificationModelPath, Mat& frame) {
    // Load detection and classification models
    Net netDetection = readNetFromONNX(detectionModelPath);
    netDetection.setPreferableBackend(DNN_BACKEND_OPENCV);
    netDetection.setPreferableTarget(DNN_TARGET_CPU);

    Net netClassification = readNetFromONNX(classificationModelPath);
    netClassification.setPreferableBackend(DNN_BACKEND_OPENCV);
    netClassification.setPreferableTarget(DNN_TARGET_CPU);

    // Preprocess the frame for detection
    Mat blob;
    blobFromImage(frame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    netDetection.setInput(blob);

    // Perform detection
    // Get all output layer names
    vector<String> outLayerNames = netDetection.getUnconnectedOutLayersNames();
    vector<Mat> outs;
    netDetection.forward(outs, outLayerNames);

    // Initialize vectors for bounding boxes, confidences, and class IDs
    vector<Rect> boxes;
    vector<float> confidences;
    
    // Parse YOLOv8 output format
    // The primary detection output should be the first one (outs[0])
    // YOLOv8 format: [num_detections, 84+] where each row has:
    // [x, y, w, h, confidence, class_scores...]
    
    // Get the first output layer (main detection results)
    const Mat& output = outs[0];
    
    float* data = (float*)output.data;
    const int dimensions = output.size[1]; // Number of values per detection
    const int numDetections = output.size[0]; // Number of detections
    
    const int coordinatesOffset = 0; // Position where box coordinates start
    const int confidenceOffset = 4; // Position of confidence score
    const int classOffset = 5;      // Position where class scores start
    const int numClasses = dimensions - classOffset; // Number of classes
    
    // Process each detection
    for (int i = 0; i < numDetections; ++i) {
        // Get pointer to the current detection data
        float* detection = &data[i * dimensions];
        
        // Get confidence
        float confidence = detection[confidenceOffset];
        
        // Find the class with highest score
        int classId = 0;
        float maxScore = 0;
        for (int j = 0; j < numClasses; ++j) {
            float score = detection[classOffset + j];
            if (score > maxScore) {
                maxScore = score;
                classId = j;
            }
        }
        
        // Filter detections by confidence threshold
        float objectConfidence = confidence * maxScore;
        if (objectConfidence > 0.5) {
            // YOLOv8 outputs centerX, centerY, width, height (normalized)
            float centerX = detection[coordinatesOffset];
            float centerY = detection[coordinatesOffset + 1];
            float width = detection[coordinatesOffset + 2];
            float height = detection[coordinatesOffset + 3];
            
            // Convert to top-left corner coordinates
            int x1 = static_cast<int>((centerX - width/2) * frame.cols);
            int y1 = static_cast<int>((centerY - height/2) * frame.rows);
            int boxWidth = static_cast<int>(width * frame.cols);
            int boxHeight = static_cast<int>(height * frame.rows);
            
            // Clamp bounding box coordinates to frame dimensions
            x1 = max(0, min(x1, frame.cols - 1));
            y1 = max(0, min(y1, frame.rows - 1));
            boxWidth = min(boxWidth, frame.cols - x1);
            boxHeight = min(boxHeight, frame.rows - y1);
            
            // Create the rect and add to vectors
            Rect diceBox(x1, y1, boxWidth, boxHeight);
            boxes.push_back(diceBox);
            confidences.push_back(objectConfidence);
        }
    }
    
    // Apply non-maximum suppression to remove overlapping boxes
    vector<int> indices;
    NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
    
    // Process surviving detections
    for (size_t i = 0; i < indices.size(); ++i) {
        int idx = indices[i];
        Rect box = boxes[idx];
        
        // Crop the detected region
        Mat cropped = frame(box).clone();
        
        // Perform classification if the cropped frame is valid
        if (!cropped.empty()) {
            // Preprocess for classification
            Mat blobClassify;
            resize(cropped, cropped, Size(224, 224));
            
            // Classify the cropped image
            Mat blob;
            blobFromImage(cropped, blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
            netClassification.setInput(blob);
            
            Mat output = netClassification.forward();
            
            // Get the predicted class
            Point classIdPoint;
            double classConfidence;
            minMaxLoc(output, 0, &classConfidence, 0, &classIdPoint);
            int classId = classIdPoint.x;
            
            // Display classification result
            rectangle(frame, box, Scalar(0, 255, 0), 2);
            putText(frame, "Dice: " + to_string(classId + 1), 
                    Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
            
            cout << "Detected dice with value: " << classId + 1 
                 << " (confidence: " << classConfidence << ")" << endl;
        }
    }

    // Display the frame with detections
    imshow("YOLO Dice Detection", frame);
}
int main(int argc, char** argv) {
    string detectionModelPath = "yolo11m_detection.onnx";
    string classificationModelPath = "yolo11s_cls.onnx";

    if (argc == 3 && string(argv[1]) == "classify") {
        Mat image = imread(argv[2]);
        if (image.empty()) {
            cout << "Error loading image: " << argv[2] << endl;
            return -1;
        }
        classifyImage(classificationModelPath, image);
        waitKey(0);
        return 0;
    }

    if (argc == 3 && string(argv[1]) == "detect") {
        Mat image = imread(argv[2]);
        if (image.empty()) {
            cout << "Error loading image: " << argv[2] << endl;
            return -1;
        }
        detectDice(detectionModelPath, classificationModelPath, image);
        waitKey(0);
        return 0;
    }

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening camera" << endl;
        return -1;
    }

    Mat frame;
    while (cap.read(frame)) {
        detectDice(detectionModelPath, classificationModelPath, frame);
        if (waitKey(1) == 27) break; // Press ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}
