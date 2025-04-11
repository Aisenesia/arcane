#include "PlayerReady.h"
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <stdexcept>

using namespace cv;
using namespace std;
using namespace std::chrono;

PlayerReady::PlayerReady(int roiRadius, Point center)
    : roiRadius(roiRadius), center(center) { 
    // roiRadius ve center, constructor ile dışarıdan alınıyor
}

void PlayerReady::setFrame(const Mat& newFrame) {
    newFrame.copyTo(frame);
}

bool PlayerReady::checkReady(int secondsToCheck) {
    if (frame.empty()) {
        throw runtime_error("Frame has not been set!");
    }

    auto startTime = high_resolution_clock::now();

    while (true) {
        if (frame.empty()) {
            throw runtime_error("Frame is empty!");
        }

        if (isGreenCovered()) {
            auto now = high_resolution_clock::now();
            auto duration = duration_cast<seconds>(now - startTime);
            if (duration.count() >= secondsToCheck) {
                return true;
            }
        } else {
            startTime = high_resolution_clock::now();
        }

        if (waitKey(1) == 27) {
            break;
        }
    }

    return false;
}

bool PlayerReady::isGreenCovered() {
    Mat mask = Mat::zeros(frame.size(), CV_8UC1);
    circle(mask, center, roiRadius, Scalar(255), FILLED);

    Mat hsvFrame;
    cvtColor(frame, hsvFrame, COLOR_BGR2HSV);

    Mat greenMask;
    inRange(hsvFrame, lowerGreen, upperGreen, greenMask);

    Mat roiGreenMask;
    bitwise_and(greenMask, mask, roiGreenMask);

    int totalPixels = countNonZero(mask);
    int greenPixels = countNonZero(roiGreenMask);
    double coverageRatio = static_cast<double>(greenPixels) / totalPixels;

    return coverageRatio <= 0.4; // Covered if green is NOT dominating
}
