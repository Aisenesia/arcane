#pragma once

#ifdef ARCANE_DLL_EXPORTS
#define ARCANE_DLL_API __declspec(dllexport)
#else
#define ARCANE_DLL_API __declspec(dllimport)
#endif

#include <opencv2/core.hpp>

// -----------------------------
// Detection & Classification
// -----------------------------
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
    ARCANE_DLL_API DetectionResultArray Detect(const cv::Mat& frame);
    ARCANE_DLL_API ClassificationResult Classify(const cv::Mat& frame);
    ARCANE_DLL_API void Cleanup();
}

// -----------------------------
// PlayerReady Integration
// -----------------------------
extern "C" {
    // Create and destroy PlayerReady instance
    ARCANE_DLL_API void* CreatePlayerReady(int roiRadius, cv::Point center);
    ARCANE_DLL_API void DestroyPlayerReady(void* instance);

    // Set frame to PlayerReady
    ARCANE_DLL_API void PlayerReady_SetFrame(void* instance, const cv::Mat& frame);

    // Check if the green area is covered long enough
    ARCANE_DLL_API bool PlayerReady_CheckReady(void* instance, int secondsToCheck);
}

// -----------------------------
// CardMovementTracker Integration (Updated)
// -----------------------------

extern "C" {
    // Create and destroy tracker
    ARCANE_DLL_API void* CreateCardMovementTracker();
    ARCANE_DLL_API void DestroyCardMovementTracker(void* instance);

    // Set frame and card list
    ARCANE_DLL_API void CardMovementTracker_SetFrame(void* instance, const cv::Mat& frame);
    ARCANE_DLL_API void CardMovementTracker_SetCardList(void* instance, const Card* cardList, int cardCount);

    // Detection logic
    ARCANE_DLL_API bool CardMovementTracker_DetectMovement(void* instance);
    ARCANE_DLL_API int CardMovementTracker_GetLastMovedCardId(void* instance);
    ARCANE_DLL_API const char* CardMovementTracker_GetLastMovedCardName(void* instance);
}