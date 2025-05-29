#include "network_manager.h"
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <windows.h>
#include <iostream>
#include <algorithm>
#include <map>

#define DICE_CLASS 0 // Define DICE_CLASS here since it's used in classifyRegion

// Static member definitions
cv::dnn::Net NetworkManager::netDetection;
cv::dnn::Net NetworkManager::netClassification;
bool NetworkManager::useCudaGlobal = false;

bool NetworkManager::checkCudaComputeCapability() {
    int deviceCount = cv::cuda::getCudaEnabledDeviceCount();
    if (deviceCount == 0) {
        std::cout << "No CUDA-enabled devices found. Falling back to CPU." << std::endl;
        return false;
    }

    for (int i = 0; i < deviceCount; ++i) {
        cv::cuda::DeviceInfo deviceInfo(i);
        int major = deviceInfo.majorVersion();
        int minor = deviceInfo.minorVersion();

        std::cout << "Device " << i << ": " << deviceInfo.name() << " (Compute Capability: "
            << major << "." << minor << ")" << std::endl;

        if (major > REQUIRED_CUDA_MAJOR || (major == REQUIRED_CUDA_MAJOR && minor >= REQUIRED_CUDA_MINOR)) {
            return true;
        }
    }

    std::cout << "No CUDA device meets the required compute capability ("
        << REQUIRED_CUDA_MAJOR << "." << REQUIRED_CUDA_MINOR << "). Falling back to CPU." << std::endl;
    return false;
}

cv::dnn::Net NetworkManager::initializeNetwork(const std::string& modelPath) {
    cv::dnn::Net net = cv::dnn::readNetFromONNX(modelPath);

    if (useCudaGlobal) {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA_FP16);
    }
    else {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }
    return net;
}

bool NetworkManager::initializeNetworks(const std::string& detectionModelPath, 
                                       const std::string& classificationModelPath, 
                                       bool useCuda) {
    if (useCuda) {
        useCudaGlobal = checkCudaComputeCapability();
    }
    
    try {
        netDetection = initializeNetwork(detectionModelPath);
        netClassification = initializeNetwork(classificationModelPath);
        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Error initializing networks: " << e.what() << std::endl;
        return false;
    }
}

void NetworkManager::cleanup() {
    netDetection = cv::dnn::Net();
    netClassification = cv::dnn::Net();
    if (useCudaGlobal) {
        cv::cuda::resetDevice();
    }
}

int NetworkManager::classConverter(int classId) {
    std::map<int, int> class_mapping = {
        {0, 1}, {1, 2}, {12, 3}, {13, 4}, {14, 5}, {15, 6},
        {16, 7}, {17, 8}, {18, 9}, {19, 10},
        {2, 11}, {3, 12}, {4, 13}, {5, 14}, {6, 15},
        {7, 16}, {8, 17}, {9, 18}, {10, 19}, {11, 20}
    };

    auto it = class_mapping.find(classId);
    return (it != class_mapping.end()) ? it->second : -1;
}

cv::Mat NetworkManager::preprocessImage(const cv::Mat& frame, const cv::Size& targetSize, bool useCuda) {
    if (useCuda && cv::cuda::getCudaEnabledDeviceCount() > 0) {
        cv::cuda::GpuMat gpuFrame, resizedGpuFrame;
        gpuFrame.upload(frame);
        cv::cuda::resize(gpuFrame, resizedGpuFrame, targetSize);
        cv::Mat resizedFrame;
        resizedGpuFrame.download(resizedFrame);
        return resizedFrame;
    }
    else {
        cv::Mat resizedFrame;
        cv::resize(frame, resizedFrame, targetSize);
        return resizedFrame;
    }
}

cv::Mat NetworkManager::scaleToFitScreen(const cv::Mat& image) {
    int screenWidth = GetSystemMetrics(SM_CXSCREEN);
    int screenHeight = GetSystemMetrics(SM_CYSCREEN);
    double scaleFactor = std::min((double)screenWidth / image.cols, (double)screenHeight / image.rows);
    cv::Mat scaledImage;
    cv::resize(image, scaledImage, cv::Size(), scaleFactor, scaleFactor);
    return scaledImage;
}

cv::Rect NetworkManager::adjustToSquare(const cv::Rect& box, int frameWidth, int frameHeight) {
    int maxEdge = std::max(box.width, box.height);
    int centerX = box.x + box.width / 2;
    int centerY = box.y + box.height / 2;

    int newLeft = std::max(0, centerX - maxEdge / 2);
    int newTop = std::max(0, centerY - maxEdge / 2);
    int newRight = std::min(frameWidth, centerX + maxEdge / 2);
    int newBottom = std::min(frameHeight, centerY + maxEdge / 2);

    return cv::Rect(newLeft, newTop, newRight - newLeft, newBottom - newTop);
}

std::vector<cv::Rect> NetworkManager::processDetections(const cv::Mat& output, const cv::Mat& frame,
                                                       std::vector<float>& confidences, std::vector<int>& classIds) {
    std::vector<cv::Rect> boxes;
    
    // Debug: Print actual output dimensions
    std::cout << "YOLO Output Debug:" << std::endl;
    std::cout << "  Dimensions: " << output.dims << std::endl;
    std::cout << "  Shape: [";
    for (int i = 0; i < output.dims; i++) {
        std::cout << output.size[i];
        if (i < output.dims - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "  Frame size: " << frame.cols << "x" << frame.rows << std::endl;
    
    int numDetections = output.size[2]; // Should be 8400
    int numAttributes = output.size[1]; // Should be 16 (4 bbox + 12 classes)
    int validDetections = 0;
    
    // YOLOv8/v11 format: [x, y, w, h, class0_prob, class1_prob, ..., class11_prob]
    // With shape [1, 16, 8400], we have 4 bbox coords + 12 class probabilities
    int numClasses = numAttributes - 4; // 16 - 4 = 12 classes
    
    for (int i = 0; i < numDetections; i++) {
        // Extract bounding box coordinates (first 4 attributes)
        float cx = output.ptr<float>(0)[0 * numDetections + i]; // x center
        float cy = output.ptr<float>(0)[1 * numDetections + i]; // y center  
        float w = output.ptr<float>(0)[2 * numDetections + i];  // width
        float h = output.ptr<float>(0)[3 * numDetections + i];  // height
        
        // Find the class with highest probability (attributes 4-15)
        int bestClassId = 0;
        float bestClassScore = 0;
        
        for (int c = 0; c < numClasses; c++) {
            float classScore = output.ptr<float>(0)[(4 + c) * numDetections + i];
            if (classScore > bestClassScore) {
                bestClassScore = classScore;
                bestClassId = c;
            }
        }
        
        // Use the highest class score as confidence
        float confidence = bestClassScore;
        
        if (confidence >= 0.5) {
            validDetections++;
            
            // Debug: Print first few detections
            if (validDetections <= 3) {
                std::cout << "  Detection " << validDetections << ": conf=" << confidence 
                         << ", cx=" << cx << ", cy=" << cy << ", w=" << w << ", h=" << h << std::endl;
                std::cout << "    Best class: " << bestClassId << " (score=" << bestClassScore << ")" << std::endl;
            }
              // Convert from 640x640 model coordinates to actual frame coordinates
            // YOLO outputs coordinates relative to 640x640 input size
            float scaleX = static_cast<float>(frame.cols) / 640.0f;
            float scaleY = static_cast<float>(frame.rows) / 640.0f;
            
            int left = static_cast<int>((cx - w / 2) * scaleX);
            int top = static_cast<int>((cy - h / 2) * scaleY);
            int width = static_cast<int>(w * scaleX);
            int height = static_cast<int>(h * scaleY);
            
            // Debug: Print converted coordinates for first few detections
            if (validDetections <= 3) {
                std::cout << "    Pixel coords: x=" << left << ", y=" << top 
                         << ", w=" << width << ", h=" << height << std::endl;
            }
            
            // Ensure bounding box is within frame boundaries
            left = std::max(0, std::min(left, frame.cols - 1));
            top = std::max(0, std::min(top, frame.rows - 1));
            width = std::min(width, frame.cols - left);
            height = std::min(height, frame.rows - top);
            
            // Only add valid detections
            if (width > 0 && height > 0) {
                boxes.push_back(cv::Rect(left, top, width, height));
                confidences.push_back(confidence);
                classIds.push_back(bestClassId);
            }
        }
    }
    
    std::cout << "  Total valid detections: " << validDetections << std::endl;
    std::cout << "  Final boxes count: " << boxes.size() << std::endl;
    
    return boxes;
}

DetectionResult NetworkManager::classifyRegion(const cv::Mat& cropped, const cv::Rect& box, 
                                              cv::dnn::Net& netClassification, int frameWidth, int frameHeight) {
    cv::Mat blob;
    cv::dnn::blobFromImage(cropped, blob, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), true, false);
    netClassification.setInput(blob);

    cv::Mat output = netClassification.forward();
    cv::Point classIdPoint;
    double confidence;
    cv::minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int classId = classConverter(classIdPoint.x);

    cv::Rect adjustedBox = (classId == DICE_CLASS) ? adjustToSquare(box, frameWidth, frameHeight) : box;
    return { adjustedBox, classId, static_cast<float>(confidence) };
}

DetectionResultArray NetworkManager::detect(const cv::Mat& frame) {
    DetectionResult* results = new DetectionResult[MAX_DETECTIONS];
    int count = 0;

    cv::Mat resizedFrame = preprocessImage(frame, cv::Size(640, 640), useCudaGlobal);

    cv::Mat blob;
    cv::dnn::blobFromImage(resizedFrame, blob, 1.0 / 255.0, cv::Size(640, 640), cv::Scalar(), true, false);
    netDetection.setInput(blob);    cv::Mat output = netDetection.forward();
    std::cout << "Detection Debug - Forward pass completed" << std::endl;
    
    if (output.dims == 3 && output.size[1] >= 5) {
        std::vector<float> confidences;
        std::vector<int> classIds;
        std::vector<cv::Rect> boxes = processDetections(output, frame, confidences, classIds);

        std::cout << "Before NMS: " << boxes.size() << " boxes" << std::endl;
        
        std::vector<int> indices;
        if (!boxes.empty()) {
            cv::dnn::NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
        }
        
        std::cout << "After NMS: " << indices.size() << " boxes" << std::endl;
        count = std::min(static_cast<int>(indices.size()), MAX_DETECTIONS);
        
        // Simply collect all detections without filtering
        for (int i = 0; i < count; i++) {
            int idx = indices[i];
            cv::Rect box = boxes[idx];
            int detectedClassId = classIds[idx];
            
            // Store detection result directly
            results[i] = { box, detectedClassId, confidences[idx] };
        }
    }

    DetectionResultArray resultArray;
    resultArray.results = results;
    resultArray.size = count;
    return resultArray;
}

ClassificationResult NetworkManager::classify(const cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(224, 224), cv::Scalar(), true, false);
    netClassification.setInput(blob);

    cv::Mat output = netClassification.forward();
    cv::Point classIdPoint;
    double confidence;
    cv::minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int classId = classConverter(classIdPoint.x);

    return { classId, static_cast<float>(confidence) };
}
