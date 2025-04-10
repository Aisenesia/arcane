#pragma once

#ifdef ARCANE_DLL_EXPORTS
#define ARCANE_DLL_API __declspec(dllexport)
#else
#define ARCANE_DLL_API __declspec(dllimport)
#endif

#include <opencv2/core.hpp>

#define DICE_CLASS 0


struct DetectionResult {
    cv::Rect boundingBox;
    int classId;
    float confidence;
};

struct DetectionResultArray {
    DetectionResult* results;
    int size;
};

struct ClassificationResult {
    int classId;
    float confidence;
};

extern "C" {
    ARCANE_DLL_API bool InitializeNetworks(const char* detectionModelPath, const char* classificationModelPath, bool useCuda);
    ARCANE_DLL_API DetectionResultArray Detect(const cv::Mat& frame); // Updated function signature
    ARCANE_DLL_API ClassificationResult Classify(const cv::Mat& frame);
    ARCANE_DLL_API void Cleanup();
}
