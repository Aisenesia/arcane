#pragma once
#include <opencv2/opencv.hpp>
#include <vector>

#define NOMINMAX

// Detection result structure
struct DetectionResult {
    cv::Rect boundingBox;
    int classId;
    float confidence;
};

// Array of detection results
struct DetectionResultArray {
    DetectionResult* results;
    int size;
    
    // Constructor
    DetectionResultArray() : results(nullptr), size(0) {}
    
    // Destructor
    ~DetectionResultArray() {
        if (results) {
            delete[] results;
            results = nullptr;
        }
    }
    
    // Copy constructor
    DetectionResultArray(const DetectionResultArray& other) : results(nullptr), size(0) {
        if (other.size > 0 && other.results) {
            size = other.size;
            results = new DetectionResult[size];
            for (int i = 0; i < size; i++) {
                results[i] = other.results[i];
            }
        }
    }
    
    // Assignment operator
    DetectionResultArray& operator=(const DetectionResultArray& other) {
        if (this != &other) {
            if (results) {
                delete[] results;
                results = nullptr;
            }
            
            if (other.size > 0 && other.results) {
                size = other.size;
                results = new DetectionResult[size];
                for (int i = 0; i < size; i++) {
                    results[i] = other.results[i];
                }
            } else {
                size = 0;
            }
        }
        return *this;
    }
};

// Classification result structure
struct ClassificationResult {
    int classId;
    float confidence;
};
