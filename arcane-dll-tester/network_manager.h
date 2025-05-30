#pragma once
#include <opencv2/opencv.hpp>
#include <opencv2/dnn.hpp>
#include "detection_types.h"

#define NOMINMAX


class NetworkManager {
public:
    // Static methods for network management
    static bool checkCudaComputeCapability();
    static bool initializeNetworks(const std::string& detectionModelPath, 
                                 const std::string& classificationModelPath, 
                                 bool useCuda = false);
    static void cleanup();
    
    // Detection and classification methods
    static DetectionResultArray detect(const cv::Mat& frame);
    static ClassificationResult classify(const cv::Mat& frame);
      // Utility methods
    static cv::Mat preprocessImage(const cv::Mat& frame, const cv::Size& targetSize, bool useCuda = false);
    static cv::Mat scaleToFitScreen(const cv::Mat& image);
    static int classConverter(int classId);
    static std::string detectionClassToName(int classId);
    
    // Getter for CUDA usage status
    static bool isUsingCuda() { return useCudaGlobal; }

private:
    static cv::dnn::Net netDetection;
    static cv::dnn::Net netClassification;
    static bool useCudaGlobal;
    
    // Constants
    static const int REQUIRED_CUDA_MAJOR = 8;
    static const int REQUIRED_CUDA_MINOR = 6;
    static const int MAX_DETECTIONS = 100;
    
    // Helper methods
    static cv::dnn::Net initializeNetwork(const std::string& modelPath);
    static cv::Rect adjustToSquare(const cv::Rect& box, int frameWidth, int frameHeight);    static std::vector<cv::Rect> processDetections(const cv::Mat& output, const cv::Mat& frame, 
                                                  std::vector<float>& confidences, std::vector<int>& classIds);
    static DetectionResult classifyRegion(const cv::Mat& cropped, const cv::Rect& box, 
                                        cv::dnn::Net& netClassification, int frameWidth, int frameHeight);
};
