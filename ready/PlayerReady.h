#ifndef PLAYER_READY_H
#define PLAYER_READY_H

#include <opencv2/opencv.hpp>
#include <chrono>

class PlayerReady {
public:
    PlayerReady(int roiRadius, cv::Point center);

    void setFrame(const cv::Mat& newFrame);
    bool checkReady(int secondsToCheck = 3);

private:
    cv::Mat frame;
    cv::Point center;
    int roiRadius;

    cv::Scalar lowerGreen = cv::Scalar(40, 50, 50);
    cv::Scalar upperGreen = cv::Scalar(80, 255, 255);

    bool isGreenCovered();
};

#endif // PLAYER_READY_H
