#include "game_components.h"
#include "network_manager.h"
#include <iostream>
#include <algorithm>
#include <stdexcept>
#include <limits>

// CalibrationChecker implementation
CalibrationChecker::CalibrationChecker(const std::vector<cv::Point>& centers, const std::vector<int>& radii) {
    if (centers.size() != 2 || radii.size() != 2) {
        throw std::invalid_argument("Exactly 2 centers and 2 radii required");
    }

    expectedCenters = centers;
    expectedRadii = radii;

    // Default green HSV range
    lowerGreen = cv::Scalar(35, 100, 100);
    upperGreen = cv::Scalar(85, 255, 255);

    // Default values
    minRadius = 10;
    centerTolerance = 20;
    radiusTolerance = 10;
}

void CalibrationChecker::setFrame(const cv::Mat& frame) {
    currentFrame = frame.clone();
}

bool CalibrationChecker::checkCalibration() {
    if (currentFrame.empty()) {
        std::cerr << "Frame not set!" << std::endl;
        return false;
    }

    // Convert to HSV
    cv::Mat hsvFrame;
    cv::cvtColor(currentFrame, hsvFrame, cv::COLOR_BGR2HSV);

    // Create green mask
    cv::Mat greenMask;
    cv::inRange(hsvFrame, lowerGreen, upperGreen, greenMask);

    bool allCirclesFilled = true;

    // Check each expected circle area for green fill
    for (size_t i = 0; i < expectedCenters.size() && i < expectedRadii.size(); ++i) {
        // Create a circular mask for this circle
        cv::Mat circleMask = cv::Mat::zeros(currentFrame.size(), CV_8UC1);
        cv::circle(circleMask, expectedCenters[i], expectedRadii[i], cv::Scalar(255), cv::FILLED);

        // Get green pixels within this circle
        cv::Mat circleGreenMask;
        cv::bitwise_and(greenMask, circleMask, circleGreenMask);

        // Count pixels
        int totalPixels = cv::countNonZero(circleMask);
        int greenPixels = cv::countNonZero(circleGreenMask);
        
        // Calculate green coverage ratio
        double greenRatio = static_cast<double>(greenPixels) / totalPixels;
        
        // Consider circle filled if more than 70% is green
        if (greenRatio < 0.7) {
            allCirclesFilled = false;
        }
    }

    return allCirclesFilled;
}

void CalibrationChecker::setGreenRange(const cv::Scalar& lower, const cv::Scalar& upper) {
    lowerGreen = lower;
    upperGreen = upper;
}

void CalibrationChecker::setTolerances(int centerTol, int radiusTol) {
    centerTolerance = centerTol;
    radiusTolerance = radiusTol;
}

// PlayerReady implementation
PlayerReady::PlayerReady(int roiRadius, cv::Point center, int secondsToWait)
    : roiRadius(roiRadius), center(center), secondsToWait(secondsToWait) {
}

void PlayerReady::setFrame(const cv::Mat& newFrame) {
    this->frame = newFrame;
}

bool PlayerReady::checkReady() {
    if (frame.empty()) {
        throw std::runtime_error("Frame has not been set!");
    }

    bool greenCovered = isGreenCovered();

    // Initially, green should be visible once
    if (!hasSeenGreen) {
        if (!greenCovered) {
            hasSeenGreen = true;
            printf("Green detected!\n");
        }
        else {
            return false;
        }
    }

    // After green has been seen, start timing when covered
    if (greenCovered) {
        if (!timerStarted) {
            startTime = std::chrono::high_resolution_clock::now();
            timerStarted = true;
            printf("Timer started!\n");
        }
        else {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime);
            double elapsedSeconds = duration.count() / 1000.0;
            
            // Calculate progress (0.0 to 1.0)
            double progress = std::min(1.0, elapsedSeconds / secondsToWait);
            
            // Draw outer circle border
            cv::circle(frame, center, roiRadius, cv::Scalar(255, 255, 255), 2);
            
            // Draw inner filled circle that grows with progress
            int innerRadius = static_cast<int>(roiRadius * progress);
            if (innerRadius > 0) {
                // Color changes from green to yellow to red as progress increases
                cv::Scalar fillColor;
                if (progress < 0.5) {
                    // Green to yellow (0.0 to 0.5)
                    fillColor = cv::Scalar(0, 255, static_cast<int>(255 * progress * 2));
                } else {
                    // Yellow to red (0.5 to 1.0)
                    fillColor = cv::Scalar(0, static_cast<int>(255 * (1.0 - progress) * 2), 255);
                }
                cv::circle(frame, center, innerRadius, fillColor, cv::FILLED);
            }
            
            // Display elapsed seconds as text above center (only show if <= 3 seconds)
            if (elapsedSeconds <= 3.0) {
                std::string timeText = std::to_string(static_cast<int>(elapsedSeconds)) + "s";
                cv::Point textPos(center.x - 15, center.y - 30); // Position above center
                
                // Draw text with background for better visibility
                cv::putText(frame, timeText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(0, 0, 0), 3);
                cv::putText(frame, timeText, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.8, cv::Scalar(255, 255, 255), 2);
            }
            
            if (elapsedSeconds >= this->secondsToWait) {
                //printf("Timer completed! Player ready!\n");
                return true;
            }
        }
    }
    else {
        if (timerStarted) {
            printf("Timer reset!\n");
        }
        timerStarted = false; // Reset timer
    }

    return false;
}

bool PlayerReady::isGreenCovered() {
    cv::Mat mask = cv::Mat::zeros(frame.size(), CV_8UC1);
    cv::circle(mask, center, roiRadius, cv::Scalar(255), cv::FILLED);

    cv::Mat hsvFrame;
    cv::cvtColor(frame, hsvFrame, cv::COLOR_BGR2HSV);

    cv::Mat greenMask;
    cv::inRange(hsvFrame, lowerGreen, upperGreen, greenMask);

    cv::Mat roiGreenMask;
    cv::bitwise_and(greenMask, mask, roiGreenMask);

    int totalPixels = cv::countNonZero(mask);
    int greenPixels = cv::countNonZero(roiGreenMask);
    double coverageRatio = static_cast<double>(greenPixels) / totalPixels;

    return coverageRatio <= 0.4; // Covered if green is NOT dominating
}
