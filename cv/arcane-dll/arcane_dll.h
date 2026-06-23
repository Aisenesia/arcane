#pragma once

#ifdef ARCANE_DLL_EXPORTS
#define ARCANE_DLL_API __declspec(dllexport)
#else
#define ARCANE_DLL_API __declspec(dllimport)
#endif

#define DICE_CLASS 0

/**
 * @brief Representation of a single object detection.
 * Uses primitive types to remain portable across DLL boundaries.
 */
struct DetectionResult {
    int x;             ///< Left bounding box coordinate
    int y;             ///< Top bounding box coordinate
    int width;         ///< Bounding box width
    int height;        ///< Bounding box height
    int classId;       ///< Detected class ID
    float confidence;  ///< Confidence score (0.0 to 1.0)
};

/**
 * @brief Wrapper struct for returning detection arrays safely.
 */
struct DetectionResultArray {
    DetectionResult* results;  ///< Pointer to dynamically allocated DetectionResult array
    int size;                  ///< Number of elements in the array
};

/**
 * @brief Representation of a classification result.
 */
struct ClassificationResult {
    int classId;       ///< Classified class ID
    float confidence;  ///< Confidence score (0.0 to 1.0)
};

extern "C" {
    /**
     * @brief Loads and initializes the ONNX neural networks.
     * @param detectionModelPath Path to the ONNX object detection model.
     * @param classificationModelPath Path to the ONNX classification model.
     * @param useCuda True to enable GPU/CUDA acceleration if compute capability matches.
     * @return True if networks initialized successfully, false otherwise.
     */
    ARCANE_DLL_API bool InitializeNetworks(const char* detectionModelPath, const char* classificationModelPath, bool useCuda);

    /**
     * @brief Performs object detection on a raw image buffer.
     * @param imageData Pointer to the raw image bytes (e.g., RGB/BGR format).
     * @param width Image width in pixels.
     * @param height Image height in pixels.
     * @param channels Number of color channels (e.g., 3 for RGB/BGR, 4 for RGBA).
     * @return DetectionResultArray containing bounding boxes and confidence.
     * @note Caller is responsible for freeing the allocated memory in DetectionResultArray by calling FreeDetectionResults.
     */
    ARCANE_DLL_API DetectionResultArray Detect(const unsigned char* imageData, int width, int height, int channels);

    /**
     * @brief Performs classification on a raw image buffer.
     * @param imageData Pointer to the raw image bytes.
     * @param width Image width in pixels.
     * @param height Image height in pixels.
     * @param channels Number of color channels.
     * @return ClassificationResult.
     */
    ARCANE_DLL_API ClassificationResult Classify(const unsigned char* imageData, int width, int height, int channels);

    /**
     * @brief Frees the memory allocated inside the DLL for detection results.
     * @param array The DetectionResultArray structure returned by Detect.
     */
    ARCANE_DLL_API void FreeDetectionResults(DetectionResultArray array);

    /**
     * @brief Releases network resources and resets the CUDA device.
     */
    ARCANE_DLL_API void Cleanup();
}
