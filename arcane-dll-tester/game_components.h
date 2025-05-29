#pragma once
#include <opencv2/opencv.hpp>
#include <vector>
#include <chrono>

#define NOMINMAX


class CalibrationChecker {
public:
    CalibrationChecker(const std::vector<cv::Point>& centers, const std::vector<int>& radii);
    
    void setFrame(const cv::Mat& frame);
    bool checkCalibration();
    void setGreenRange(const cv::Scalar& lower, const cv::Scalar& upper);
    void setTolerances(int centerTol, int radiusTol);

private:
    std::vector<cv::Point> expectedCenters;
    std::vector<int> expectedRadii;
    cv::Mat currentFrame;
    
    // HSV range for green color detection
    cv::Scalar lowerGreen;
    cv::Scalar upperGreen;
    
    // Detection parameters
    int minRadius;
    int centerTolerance;
    int radiusTolerance;
};

class PlayerReady {
public:
    PlayerReady(int roiRadius, cv::Point center, int secondsToWait);
    
    void setFrame(const cv::Mat& newFrame);
    bool checkReady();

private:
    cv::Mat frame;
    int roiRadius;
    cv::Point center;
    int secondsToWait;
    
    // State tracking
    bool hasSeenGreen = false;
    bool timerStarted = false;
    std::chrono::high_resolution_clock::time_point startTime;

    // Green color detection range (HSV)
    cv::Scalar lowerGreen = cv::Scalar(35, 100, 100);
    cv::Scalar upperGreen = cv::Scalar(85, 255, 255);
    
    bool isGreenCovered();
};

