#ifndef PLAYER_READY_H
#define PLAYER_READY_H

#include <opencv2/opencv.hpp>
#include <chrono>

class PlayerReady {
public:
    PlayerReady(int roiRadius, cv::Point center);

    void setFrame(const cv::Mat& newFrame);
    bool checkReady(int secondsToCheck); // her karede çağrılır, ama zaman içinde karar verir

private:
    int roiRadius;
    cv::Point center;
    cv::Mat frame;

    bool hasSeenGreen = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    bool timerStarted = false;

    cv::Scalar lowerGreen = cv::Scalar(35, 50, 50);
    cv::Scalar upperGreen = cv::Scalar(85, 255, 255);

    bool isGreenCovered();
};

#endif // PLAYER_READY_H
