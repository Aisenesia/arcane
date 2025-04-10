#include "pch.h"
#include "arcane_dll.h"
#include <algorithm>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <windows.h>
#include <vector>


using namespace cv;
using namespace cv::dnn;
using namespace std;

Net netDetection;
Net netClassification;
bool useCudaGlobal = false;

// Define CUDA major and minor versions
const int REQUIRED_CUDA_MAJOR = 8;
const int REQUIRED_CUDA_MINOR = 6;

bool checkCudaComputeCapability() {
    int deviceCount = cuda::getCudaEnabledDeviceCount();
    if (deviceCount == 0) {
        cout << "No CUDA-enabled devices found. Falling back to CPU." << endl;
        return false;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cuda::DeviceInfo deviceInfo(i);
        int major = deviceInfo.majorVersion();
        int minor = deviceInfo.minorVersion();

        cout << "Device " << i << ": " << deviceInfo.name() << " (Compute Capability: "
            << major << "." << minor << ")" << endl;

        if (major > REQUIRED_CUDA_MAJOR || (major == REQUIRED_CUDA_MAJOR && minor >= REQUIRED_CUDA_MINOR)) {
            return true;
        }
    }

    cout << "No CUDA device meets the required compute capability ("
        << REQUIRED_CUDA_MAJOR << "." << REQUIRED_CUDA_MINOR << "). Falling back to CPU." << endl;
    return false;
}

int classConverter(int classId) {
    map<int, int> class_mapping = {
        {0, 1}, {1, 2}, {12, 3}, {13, 4}, {14, 5}, {15, 6},
        {16, 7}, {17, 8}, {18, 9}, {19, 10},
        {2, 11}, {3, 12}, {4, 13}, {5, 14}, {6, 15},
        {7, 16}, {8, 17}, {9, 18}, {10, 19}, {11, 20}
    };

    if (class_mapping.find(classId) != class_mapping.end()) {
        return class_mapping[classId];
    }
    else {
        return -1;
    }
}

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

Net initializeNetwork(const string& modelPath, bool useCuda) {
    Net net = readNetFromONNX(modelPath);
    if (useCuda && cuda::getCudaEnabledDeviceCount() > 0) {
        net.setPreferableBackend(DNN_BACKEND_CUDA);
        net.setPreferableTarget(DNN_TARGET_CUDA_FP16);
    }
    else {
        net.setPreferableBackend(DNN_BACKEND_OPENCV);
        net.setPreferableTarget(DNN_TARGET_CPU);
    }
    return net;
}

Mat preprocessImage(const Mat& frame, const Size& targetSize, bool useCuda) {
    if (useCuda && cuda::getCudaEnabledDeviceCount() > 0) {
        cuda::GpuMat gpuFrame, resizedGpuFrame;
        gpuFrame.upload(frame);
        cuda::resize(gpuFrame, resizedGpuFrame, targetSize);
        Mat resizedFrame;
        resizedGpuFrame.download(resizedFrame);
        return resizedFrame;
    }
    else {
        Mat resizedFrame;
        resize(frame, resizedFrame, targetSize);
        return resizedFrame;
    }
}

Mat scaleToFitScreen(const Mat& image) {
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    double scaleFactor = min((double)screenWidth / image.cols, (double)screenHeight / image.rows);
    Mat scaledImage;
    resize(image, scaledImage, Size(), scaleFactor, scaleFactor);
    return scaledImage;
}

// Helper function to adjust bounding box to be square
Rect adjustToSquare(const Rect& box, int frameWidth, int frameHeight) {
    int maxEdge = max(box.width, box.height);
    int centerX = box.x + box.width / 2;
    int centerY = box.y + box.height / 2;

    int newLeft = max(0, centerX - maxEdge / 2);
    int newTop = max(0, centerY - maxEdge / 2);
    int newRight = min(frameWidth, centerX + maxEdge / 2);
    int newBottom = min(frameHeight, centerY + maxEdge / 2);

    return Rect(newLeft, newTop, newRight - newLeft, newBottom - newTop);
}

// Helper function to process detection output
vector<Rect> processDetections(const Mat& output, const Mat& frame, vector<float>& confidences) {
    vector<Rect> boxes;
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
                boxes.push_back(Rect(left, top, width, height));
                confidences.push_back(confidence);
            }
        }
    }
    return boxes;
}

// Helper function to classify cropped regions
DetectionResult classifyRegion(const Mat& cropped, const Rect& box, Net& netClassification, int frameWidth, int frameHeight) {
    Mat blob;
    blobFromImage(cropped, blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
    netClassification.setInput(blob);

    Mat output = netClassification.forward();
    Point classIdPoint;
    double confidence;
    minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int classId = classConverter(classIdPoint.x);

    Rect adjustedBox = (classId == DICE_CLASS) ? adjustToSquare(box, frameWidth, frameHeight) : box;
    return { adjustedBox, classId, static_cast<float>(confidence) };
}

vector<DetectionResult> runDetection(Net& netDetection, Net& netClassification, const Mat& frame, bool useCuda) {
    vector<DetectionResult> results;
    Mat resizedFrame = preprocessImage(frame, Size(640, 640), useCuda);

    Mat blob;
    blobFromImage(resizedFrame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    netDetection.setInput(blob);

    Mat output = netDetection.forward();
    if (output.dims == 3 && output.size[1] == 5) {
        vector<float> confidences;
        vector<Rect> boxes = processDetections(output, frame, confidences);

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

                // Adjust bounding box if the class is DICE_CLASS
                Rect adjustedBox = (classId == DICE_CLASS) ? adjustToSquare(box, frame.cols, frame.rows) : box;

                results.push_back({ adjustedBox, classId, static_cast<float>(confidence) });
            }
        }
    }
    return results;
}

vector<DetectionResult> detectDice(Net& netDetection, Net& netClassification, const Mat& frame, bool useCuda) {
    vector<DetectionResult> results;
    Mat resizedFrame = preprocessImage(frame, Size(640, 640), useCuda);

    Mat blob;
    blobFromImage(resizedFrame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    netDetection.setInput(blob);

    Mat output = netDetection.forward();
    if (output.dims == 3 && output.size[1] == 5) {
        vector<float> confidences;
        vector<Rect> boxes = processDetections(output, frame, confidences);

        vector<int> indices;
        if (!boxes.empty()) {
            NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
        }

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];

            Mat cropped = frame(box).clone();
            if (!cropped.empty()) {
                results.push_back(classifyRegion(cropped, box, netClassification, frame.cols, frame.rows));
            }
        }
    }
    return results;
}

Mat generateFrame(const Mat& image, const vector<Rect>& boxes, const vector<float>& confidences) {
    Mat result = image.clone();
    for (size_t i = 0; i < boxes.size(); ++i) {
        rectangle(result, boxes[i], Scalar(0, 255, 0), 2);
        putText(result, to_string(confidences[i]), boxes[i].tl(), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
    }
    return result;
}

extern "C" ARCANE_DLL_API bool InitializeNetworks(const char* detectionModelPath, const char* classificationModelPath, bool useCuda) {
    useCudaGlobal = useCuda;
    netDetection = initializeNetwork(detectionModelPath, useCuda);
    netClassification = initializeNetwork(classificationModelPath, useCuda);
    return true;
}

extern "C" ARCANE_DLL_API DetectionResultArray Detect(const cv::Mat& frame) {
    const int maxDetections = 100; // Maximum number of detections
    DetectionResult* results = new DetectionResult[maxDetections];
    int count = 0;

    Mat resizedFrame = preprocessImage(frame, Size(640, 640), useCudaGlobal);

    Mat blob;
    blobFromImage(resizedFrame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);
    netDetection.setInput(blob);

    Mat output = netDetection.forward();
    if (output.dims == 3 && output.size[1] == 5) {
        
        vector<float> confidences;

        vector<Rect> boxes = processDetections(output, frame, confidences);
        count = static_cast<int>(boxes.size());

        for (int i = 0; i < count; i++) {
            Mat cropped = frame(boxes[i]).clone();
            if (!cropped.empty()) {
                results[i] = classifyRegion(cropped, boxes[i], netClassification, frame.cols, frame.rows);
            }
        }
    }

    DetectionResultArray resultArray;
    resultArray.results = results;
    resultArray.size = count;
    return resultArray;
}

extern "C" ARCANE_DLL_API ClassificationResult Classify(const Mat& frame) {
    Mat blob;
    blobFromImage(frame, blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
    netClassification.setInput(blob);

    Mat output = netClassification.forward();
    Point classIdPoint;
    double confidence;
    minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int classId = classConverter(classIdPoint.x);

    return { classId, static_cast<float>(confidence) };
}

extern "C" ARCANE_DLL_API void Cleanup() {
    netDetection = Net(); // Release the detection network
    netClassification = Net(); // Release the classification network
    cuda::resetDevice(); // Reset the CUDA device to release all resources
}