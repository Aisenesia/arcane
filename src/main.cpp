#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include <iostream>
#include <algorithm>

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
    // Load detection model
    Net netDetection = readNetFromONNX(detectionModelPath);
    netDetection.setPreferableBackend(DNN_BACKEND_OPENCV);
    netDetection.setPreferableTarget(DNN_TARGET_CPU);
    
    // Load classification model
    Net netClassification = readNetFromONNX(classificationModelPath);
    netClassification.setPreferableBackend(DNN_BACKEND_OPENCV);
    netClassification.setPreferableTarget(DNN_TARGET_CPU);

    // Preprocess the frame for detection
    Mat blob;
    blobFromImage(frame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    netDetection.setInput(blob);

    // Get detection output
    Mat output = netDetection.forward();
    
    cout << "Detection output shape: ";
    for (int d = 0; d < output.dims; d++) {
        cout << output.size[d] << " ";
    }
    cout << endl;
    
    // For YOLOv8/YOLO11 format with shape [1, 5, 8400]
    // First dimension: batch size (1)
    // Second dimension: values per box (x, y, w, h, confidence)
    // Third dimension: number of detections/grid cells (8400)
    
    // Ensure we have the expected format
    if (output.dims == 3 && output.size[1] == 5) {
        vector<Rect> boxes;
        vector<float> confidences;
        
        // Access the output data - shape is [1, 5, 8400]
        // In OpenCV, we need to process this correctly
        int batch = 0; // We only have one batch
        const int numDetections = output.size[2]; // 8400 detections
        const int valuesPerBox = output.size[1]; // 5 values per box
        
        cout << "Processing " << numDetections << " potential detections" << endl;
        
        // YOLO11 format:
        // output[0][0][i] = x center
        // output[0][1][i] = y center
        // output[0][2][i] = width
        // output[0][3][i] = height
        // output[0][4][i] = confidence
        
        // Access values for each detection
        for (int i = 0; i < numDetections; i++) {
            float confidence = output.ptr<float>(batch)[4 * numDetections + i];
            
            if (confidence > 0.5) { // Confidence threshold
                float cx = output.ptr<float>(batch)[0 * numDetections + i];
                float cy = output.ptr<float>(batch)[1 * numDetections + i];
                float w = output.ptr<float>(batch)[2 * numDetections + i];
                float h = output.ptr<float>(batch)[3 * numDetections + i];
                
                // Convert to corner coordinates
                int left = static_cast<int>((cx - w/2) * frame.cols);
                int top = static_cast<int>((cy - h/2) * frame.rows);
                int width = static_cast<int>(w * frame.cols);
                int height = static_cast<int>(h * frame.rows);
                
                // Ensure coordinates are valid
                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                width = min(width, frame.cols - left);
                height = min(height, frame.rows - top);
                
                // Create rectangle and add to vectors
                if (width > 0 && height > 0) {
                    Rect box(left, top, width, height);
                    boxes.push_back(box);
                    confidences.push_back(confidence);
                    
                    cout << "Found box: " << box << " with confidence: " << confidence << endl;
                }
            }
        }
        
        // Apply non-maximum suppression
        vector<int> indices;
        if (!boxes.empty()) {
            NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
            cout << "After NMS: " << indices.size() << " boxes remain" << endl;
        }
        
        // Process surviving detections
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];
            
            // Ensure the box is valid
            if (box.width <= 0 || box.height <= 0 || 
                box.x < 0 || box.y < 0 ||
                box.x + box.width > frame.cols || 
                box.y + box.height > frame.rows) {
                continue;
            }
            
            // Crop the detected region
            Mat cropped;
            try {
                cropped = frame(box).clone();
            } catch (const cv::Exception& e) {
                cerr << "Error cropping frame: " << e.what() << endl;
                continue;
            }
            
            // Perform classification if the cropped frame is valid
            if (!cropped.empty()) {
                // Preprocess for classification
                Mat resized;
                resize(cropped, resized, Size(224, 224));
                
                // Classify the cropped image
                Mat blobClassify;
                blobFromImage(resized, blobClassify, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
                netClassification.setInput(blobClassify);
                
                Mat classOutput = netClassification.forward();
                
                // Get the predicted class
                Point classIdPoint;
                double classConfidence;
                minMaxLoc(classOutput, 0, &classConfidence, 0, &classIdPoint);
                int classId = classIdPoint.x;
                
                // Draw the bounding box and class label
                rectangle(frame, box, Scalar(0, 255, 0), 2);
                putText(frame, "Dice: " + to_string(classId + 1), 
                        Point(box.x, box.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
                
                cout << "Classified dice with value: " << classId + 1 
                     << " (confidence: " << classConfidence << ")" << endl;
            }
        }
    } else {
        cout << "Unexpected output format. Expected [1, 5, N] but got a different shape." << endl;
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

void debugModelOutput(Net& net, const Mat& frame) {
    // Get input and output layer names
    vector<String> outLayerNames = net.getUnconnectedOutLayersNames();
    
    cout << "Model has " << outLayerNames.size() << " output layers:" << endl;
    for (const auto& name : outLayerNames) {
        cout << "Layer: " << name << endl;
    }
    
    // Prepare input
    Mat blob;
    blobFromImage(frame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    net.setInput(blob);
    
    // Get outputs
    vector<Mat> outs;
    net.forward(outs, outLayerNames);
    
    // Print output shapes and some values
    for (size_t i = 0; i < outs.size(); i++) {
        const Mat& output = outs[i];
        cout << "Output " << i << " shape: ";
        for (int d = 0; d < output.dims; d++) {
            cout << output.size[d] << " ";
        }
        cout << endl;
        
        // Print first few values if it's a reasonable size
        if (output.total() > 0 && output.dims <= 2) {
            cout << "First values: ";
            const float* data = (float*)output.data;
            for (int j = 0; j < min(10, (int) output.total()); j++) {
                cout << data[j] << " ";
            }
            cout << endl;
        }
    }
}
