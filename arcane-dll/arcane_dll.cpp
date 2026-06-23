#include "pch.h"
#include "arcane_dll.h"
#include <algorithm>
#include <iostream>
#include <map>
#include <vector>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>

using namespace std;

// Global network instances
cv::dnn::Net netDetection;
cv::dnn::Net netClassification;
bool useCudaGlobal = false;

// Convert raw classification ID to output class
int classConverter(int classId) {
    static const map<int, int> class_mapping = {
        {0, 1}, {1, 2}, {12, 3}, {13, 4}, {14, 5}, {15, 6},
        {16, 7}, {17, 8}, {18, 9}, {19, 10},
        {2, 11}, {3, 12}, {4, 13}, {5, 14}, {6, 15},
        {7, 16}, {8, 17}, {9, 18}, {10, 19}, {11, 20}
    };

    auto it = class_mapping.find(classId);
    return (it != class_mapping.end()) ? it->second : -1;
}

// Reconstruct cv::Mat wrapping the raw image data pointer
cv::Mat reconstructMat(const unsigned char* imageData, int width, int height, int channels) {
    if (imageData == nullptr || width <= 0 || height <= 0 || channels <= 0) {
        return cv::Mat();
    }
    int type = CV_8UC3;
    if (channels == 1) type = CV_8UC1;
    else if (channels == 4) type = CV_8UC4;
    
    // Create view and clone to ensure deep copy / memory safety
    return cv::Mat(height, width, type, const_cast<unsigned char*>(imageData)).clone();
}

// Preprocess frame by resizing on CPU (avoids slow CPU-GPU memory transfers for resize)
cv::Mat preprocessImage(const cv::Mat& frame, const cv::Size& targetSize) {
    if (frame.empty()) return cv::Mat();
    cv::Mat resizedFrame;
    cv::resize(frame, resizedFrame, targetSize);
    return resizedFrame;
}

// Helper function to adjust bounding box to be square
cv::Rect adjustToSquare(const cv::Rect& box, int frameWidth, int frameHeight) {
    int maxEdge = max(box.width, box.height);
    int centerX = box.x + box.width / 2;
    int centerY = box.y + box.height / 2;

    int newLeft = max(0, centerX - maxEdge / 2);
    int newTop = max(0, centerY - maxEdge / 2);
    int newRight = min(frameWidth, centerX + maxEdge / 2);
    int newBottom = min(frameHeight, centerY + maxEdge / 2);

    return cv::Rect(newLeft, newTop, newRight - newLeft, newBottom - newTop);
}

// Process network output detections
vector<cv::Rect> processDetections(const cv::Mat& output, const cv::Mat& frame, vector<float>& confidences) {
    vector<cv::Rect> boxes;
    if (output.dims < 3) return boxes;
    
    int numDetections = output.size[2];

    for (int i = 0; i < numDetections; i++) {
        float confidence = output.ptr<float>(0)[4 * numDetections + i];
        if (confidence > 0.5f) {
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
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    return boxes;
}

// Classify cropped region of interest
DetectionResult classifyRegion(const cv::Mat& cropped, const cv::Rect& box, cv::dnn::Net& netClassification, int frameWidth, int frameHeight) {
    cv::Mat blob;
    cv::dnn::blobFromImage(cropped, blob, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), true, false);
    netClassification.setInput(blob);

    cv::Mat output = netClassification.forward();
    cv::Point classIdPoint;
    double confidence;
    cv::minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int classId = classConverter(classIdPoint.x);

    cv::Rect adjustedBox = (classId == DICE_CLASS) ? adjustToSquare(box, frameWidth, frameHeight) : box;
    return { adjustedBox.x, adjustedBox.y, adjustedBox.width, adjustedBox.height, classId, static_cast<float>(confidence) };
}

// Initialize single network configuration
cv::dnn::Net initializeNetwork(const string& modelPath, bool useCuda) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    if (useCuda) {
        try {
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
        }
        catch (const cv::Exception& e) {
            cerr << "Warning: CUDA backend not supported by OpenCV binary. Falling back to CPU. Details: " << e.what() << endl;
            net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
            net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        }
    }
    else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return net;
}

extern "C" {

ARCANE_DLL_API bool InitializeNetworks(const char* detectionModelPath, const char* classificationModelPath, bool useCuda) {
    if (detectionModelPath == nullptr || classificationModelPath == nullptr) {
        cerr << "Error: Model paths cannot be null." << endl;
        return false;
    }

    try {
        useCudaGlobal = useCuda;
        netDetection = initializeNetwork(detectionModelPath, useCuda);
        netClassification = initializeNetwork(classificationModelPath, useCuda);
        return true;
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Exception in InitializeNetworks: " << e.what() << endl;
        return false;
    }
    catch (const std::exception& e) {
        cerr << "Standard Exception in InitializeNetworks: " << e.what() << endl;
        return false;
    }
    catch (...) {
        cerr << "Unknown Exception in InitializeNetworks" << endl;
        return false;
    }
}

ARCANE_DLL_API DetectionResultArray Detect(const unsigned char* imageData, int width, int height, int channels) {
    DetectionResultArray resultArray;
    resultArray.results = nullptr;
    resultArray.size = 0;

    if (netDetection.empty() || netClassification.empty()) {
        cerr << "Error: Networks are not initialized." << endl;
        return resultArray;
    }

    try {
        cv::Mat frame = reconstructMat(imageData, width, height, channels);
        if (frame.empty()) {
            cerr << "Error: Invalid image data." << endl;
            return resultArray;
        }

        cv::Mat resizedFrame = preprocessImage(frame, cv::Size(640, 640));
        cv::Mat blob;
        cv::dnn::blobFromImage(resizedFrame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
        netDetection.setInput(blob);

        cv::Mat output = netDetection.forward();
        if (output.dims == 3 && output.size[1] == 5) {
            vector<float> confidences;
            vector<cv::Rect> boxes = processDetections(output, frame, confidences);

            vector<int> indices;
            if (!boxes.empty()) {
                cv::dnn::NMSBoxes(boxes, confidences, 0.5f, 0.4f, indices);
            }

            int finalCount = static_cast<int>(indices.size());
            if (finalCount > 0) {
                // Dynamically allocate memory matching exactly the number of detections
                DetectionResult* results = new DetectionResult[finalCount];
                int validCount = 0;

                for (int i = 0; i < finalCount; i++) {
                    int idx = indices[i];
                    cv::Rect box = boxes[idx];

                    cv::Mat cropped = frame(box).clone();
                    if (!cropped.empty()) {
                        results[validCount] = classifyRegion(cropped, box, netClassification, frame.cols, frame.rows);
                        validCount++;
                    }
                }

                resultArray.results = results;
                resultArray.size = validCount;
            }
        }
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Exception in Detect: " << e.what() << endl;
    }
    catch (const std::exception& e) {
        cerr << "Standard Exception in Detect: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown Exception in Detect" << endl;
    }

    return resultArray;
}

ARCANE_DLL_API ClassificationResult Classify(const unsigned char* imageData, int width, int height, int channels) {
    ClassificationResult result = { -1, 0.0f };

    if (netClassification.empty()) {
        cerr << "Error: Classification network is not initialized." << endl;
        return result;
    }

    try {
        cv::Mat frame = reconstructMat(imageData, width, height, channels);
        if (frame.empty()) {
            cerr << "Error: Invalid image data." << endl;
            return result;
        }

        cv::Mat blob;
        cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), true, false);
        netClassification.setInput(blob);

        cv::Mat output = netClassification.forward();
        cv::Point classIdPoint;
        double confidence;
        cv::minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
        int classId = classConverter(classIdPoint.x);

        result.classId = classId;
        result.confidence = static_cast<float>(confidence);
    }
    catch (const cv::Exception& e) {
        cerr << "OpenCV Exception in Classify: " << e.what() << endl;
    }
    catch (const std::exception& e) {
        cerr << "Standard Exception in Classify: " << e.what() << endl;
    }
    catch (...) {
        cerr << "Unknown Exception in Classify" << endl;
    }

    return result;
}

ARCANE_DLL_API void FreeDetectionResults(DetectionResultArray array) {
    if (array.results != nullptr) {
        delete[] array.results;
    }
}

ARCANE_DLL_API void Cleanup() {
    try {
        netDetection = cv::dnn::Net();
        netClassification = cv::dnn::Net();
    }
    catch (...) {
        // Suppress errors during cleanup
    }
}

} // extern "C"