#include <algorithm>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <windows.h> // For GetSystemMetrics

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Function to get cuda status
bool getCudaStatus() {
    return cuda::getCudaEnabledDeviceCount() > 0;
}


// Function to convert class IDs
int classConverter(int classId) {
    map<int, int> class_mapping = {
        {0, 1}, {1, 2}, {12, 3}, {13, 4}, {14, 5}, {15, 6},
        {16, 7}, {17, 8}, {18, 9}, {19, 10},
        {2, 11}, {3, 12}, {4, 13}, {5, 14}, {6, 15},
        {7, 16}, {8, 17}, {9, 18}, {10, 19}, {11, 20}
    };

    if (class_mapping.find(classId) != class_mapping.end()) {
        return class_mapping[classId];
    } else {
        return -1;
    }
}

// Function to classify a cropped image
void classifyImage(Net& netClassification, const Mat& image) {
    Mat blob;
    blobFromImage(image, blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
    netClassification.setInput(blob);

    Mat output = netClassification.forward();

    Point classIdPoint;
    double confidence;
    minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int classId = classConverter(classIdPoint.x);

    cout << "Classified as: " << classId << " with confidence: " << confidence << endl;
}

// Function to initialize a network
Net initializeNetwork(const string& modelPath, bool useCuda) {
    Net net = readNetFromONNX(modelPath);
    if (useCuda && cuda::getCudaEnabledDeviceCount() > 0) {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
    } else {
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    return net;
}

// Function to preprocess an image (supports both CPU and CUDA)
Mat preprocessImage(const Mat& frame, const Size& targetSize, bool useCuda) {
    if (useCuda && cuda::getCudaEnabledDeviceCount() > 0) {
        cuda::GpuMat gpuFrame, resizedGpuFrame;
        gpuFrame.upload(frame);
        cuda::resize(gpuFrame, resizedGpuFrame, targetSize);
        Mat resizedFrame;
        resizedGpuFrame.download(resizedFrame);
        return resizedFrame;
    } else {
        Mat resizedFrame;
        resize(frame, resizedFrame, targetSize);
        return resizedFrame;
    }
}

// Function to scale an image to fit the screen
Mat scaleToFitScreen(const Mat& image) {
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    double scaleFactor = min((double)screenWidth / image.cols, (double)screenHeight / image.rows);
    Mat scaledImage;
    resize(image, scaledImage, Size(), scaleFactor, scaleFactor);
    return scaledImage;
}

// Function to detect dice in a frame
Mat detectDice(Net& netDetection, Net& netClassification, const Mat& frame, bool useCuda) {
    Mat processedFrame = frame.clone();
    Mat resizedFrame = preprocessImage(processedFrame, Size(640, 640), useCuda);

    Mat blob;
    blobFromImage(resizedFrame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    netDetection.setInput(blob);

    Mat output = netDetection.forward();
    if (output.dims == 3 && output.size[1] == 5) {
        vector<Rect> boxes;
        vector<float> confidences;

        int numDetections = output.size[2];
        for (int i = 0; i < numDetections; i++) {
            float confidence = output.ptr<float>(0)[4 * numDetections + i];
            if (confidence > 0.5) {
                float cx = output.ptr<float>(0)[0 * numDetections + i];
                float cy = output.ptr<float>(0)[1 * numDetections + i];
                float w = output.ptr<float>(0)[2 * numDetections + i];
                float h = output.ptr<float>(0)[3 * numDetections + i];

                int left = static_cast<int>((cx - w / 2) * frame.cols / 640);
                int top = static_cast<int>((cy - h / 2) * frame.rows / 640);
                int width = static_cast<int>(w * frame.cols / 640);
                int height = static_cast<int>(h * frame.rows / 640);

                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                width = min(width, frame.cols - left);
                height = min(height, frame.rows - top);

                if (width > 0 && height > 0) {
                    Rect box(left, top, width, height);
                    boxes.push_back(box);
                    confidences.push_back(confidence);
                }
            }
        }

        vector<int> indices;
        if (!boxes.empty()) {
            NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
        }

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];

            Mat cropped = frame(box).clone();
            if (!cropped.empty()) {
                Mat blob;
                blobFromImage(cropped, blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
                netClassification.setInput(blob);

                Mat output = netClassification.forward();
                Point classIdPoint;
                double confidence;
                minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
                int classId = classConverter(classIdPoint.x);

                rectangle(processedFrame, box, Scalar(0, 255, 0), 2);
                double fontScale = max(0.5, box.height / 100.0);
                int thickness = max(1, static_cast<int>(fontScale));
                putText(processedFrame, "Class " + to_string(classId), Point(box.x, box.y - 10), 
                        FONT_HERSHEY_SIMPLEX, fontScale, Scalar(0, 255, 0), thickness);
            }
        }
    }
    return processedFrame;
}

// Function to handle detection mode
void runDetectionMode(Net& netDetection, Net& netClassification, const string& imagePath, bool useCuda) {
    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "Error loading image: " << imagePath << endl;
        return;
    }
    Mat result = detectDice(netDetection, netClassification, image, useCuda);
    Mat scaledResult = scaleToFitScreen(result);
    imshow("YOLO Dice Detection", scaledResult);
    waitKey(0);
}

// Function to handle classification mode
void runClassificationMode(Net& netClassification, const string& imagePath) {
    Mat image = imread(imagePath);
    if (image.empty()) {
        cout << "Error loading image: " << imagePath << endl;
        return;
    }
    classifyImage(netClassification, image);
}

// Function to handle live detection mode
void runLiveDetectionMode(Net& netDetection, Net& netClassification, bool useCuda) {
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening camera" << endl;
        return;
    }

    Mat frame;
    while (cap.read(frame)) {
        Mat result = detectDice(netDetection, netClassification, frame, useCuda);
        Mat scaledResult = scaleToFitScreen(result);
        imshow("YOLO Dice Detection", scaledResult);
        if (waitKey(1) == 27) break;  // Press ESC to exit
    }

    cap.release();
    destroyAllWindows();
}

// Main function
int main(int argc, char** argv) {
    cout << "Arcane Dice Detection" << endl;
    string detectionModelPath = "yolov8m_detection.onnx";
    string classificationModelPath = "yolov8s_cls.onnx";

    bool useCuda = getCudaStatus(); // Set to false to force CPU fallback

    // Initialize networks
    Net netDetection = initializeNetwork(detectionModelPath, useCuda);
    Net netClassification = initializeNetwork(classificationModelPath, useCuda);

    // Parse command-line arguments
    if (argc == 3 && string(argv[1]) == "detection") {
        runDetectionMode(netDetection, netClassification, argv[2], useCuda);
    } else if (argc == 3 && string(argv[1]) == "classification") {
        runClassificationMode(netClassification, argv[2]);
    } else if (argc == 2 && string(argv[1]) == "live") {
        runLiveDetectionMode(netDetection, netClassification, useCuda);
    } else {
        cout << "Usage:" << endl;
        cout << "  " << argv[0] << " detection <image_path>       # Run detection on an image" << endl;
        cout << "  " << argv[0] << " classification <image_path> # Run classification on an image" << endl;
        cout << "  " << argv[0] << " live                        # Run live detection using webcam" << endl;
    }

    return 0;
}

