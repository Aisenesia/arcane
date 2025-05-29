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
    cv::Mat mask;
    cv::inRange(hsvFrame, lowerGreen, upperGreen, mask);

    // Detect circles using HoughCircles
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1,
        mask.rows / 8,
        100, 30,
        minRadius, minRadius * 100);

    if (circles.size() < 2) {
        return false;
    }

    // Sort circles by radius (largest first)
    std::sort(circles.begin(), circles.end(),
        [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] > b[2]; });

    // Take the top 2 circles
    std::vector<cv::Vec3f> topCircles(circles.begin(), 
        circles.begin() + std::min(2, static_cast<int>(circles.size())));

    bool allAligned = true;

    // Check each expected center
    for (size_t i = 0; i < expectedCenters.size(); ++i) {
        float minDist = std::numeric_limits<float>::max();
        int bestMatch = -1;

        for (size_t j = 0; j < topCircles.size(); ++j) {
            float dx = expectedCenters[i].x - topCircles[j][0];
            float dy = expectedCenters[i].y - topCircles[j][1];
            float dist = std::sqrt(dx * dx + dy * dy);

            if (dist < minDist) {
                minDist = dist;
                bestMatch = j;
            }
        }

        if (bestMatch >= 0) {
            // Check center distance
            if (minDist > centerTolerance) {
                std::cout << "Circle #" << i << " center out of tolerance. Distance: " << minDist << std::endl;
                allAligned = false;
            }

            // Check radius
            float radiusDiff = std::abs(expectedRadii[i] - topCircles[bestMatch][2]);
            if (radiusDiff > radiusTolerance) {
                std::cout << "Circle #" << i << " radius out of tolerance. Diff: " << radiusDiff << std::endl;
                allAligned = false;
            }
        }
        else {
            std::cout << "No match found for circle #" << i << std::endl;
            allAligned = false;
        }
    }

    return allAligned;
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
    newFrame.copyTo(frame);
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
            printf("Timer started! Filling circle...\n");
        }
        else {
            auto now = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime);
            double elapsedSeconds = duration.count() / 1000.0;
            
            // Draw progress circle
            drawProgressCircle(elapsedSeconds);
            
            if (elapsedSeconds >= this->secondsToWait) {
                printf("Timer completed! Player ready!\n");
                return true;
            }
        }
    }
    else {
        if (timerStarted) {
            printf("Timer reset! Clearing circle fill...\n");
        }
        timerStarted = false; // Reset timer
        
    }

    return false;
}


void PlayerReady::drawProgressCircle(double elapsedSeconds) {
    if (frame.empty()) return;
    
    // Calculate progress percentage (0.0 to 1.0)
    double progress = std::min(1.0, elapsedSeconds / secondsToWait);
    
    // Calculate the angle for the arc (0 to 360 degrees)
    int endAngle = static_cast<int>(progress * 360);
    
    // Draw the background circle (gray)
    cv::circle(frame, center, roiRadius, cv::Scalar(128, 128, 128), 3);
    
    // Draw the progress arc (green to red gradient based on progress)
    if (endAngle > 0) {
        // Color changes from green to yellow to red as progress increases
        cv::Scalar progressColor;
        if (progress < 0.5) {
            // Green to yellow (0.0 to 0.5)
            progressColor = cv::Scalar(0, 255, static_cast<int>(255 * progress * 2));
        } else {
            // Yellow to red (0.5 to 1.0)
            progressColor = cv::Scalar(0, static_cast<int>(255 * (1.0 - progress) * 2), 255);
        }
        
        // Draw filled sector (pie slice)
        std::vector<cv::Point> arcPoints;
        arcPoints.push_back(center);
        
        // Generate points for the arc
        for (int angle = -90; angle <= endAngle - 90; angle += 2) {
            double radians = angle * CV_PI / 180.0;
            int x = center.x + static_cast<int>(roiRadius * cos(radians));
            int y = center.y + static_cast<int>(roiRadius * sin(radians));
            arcPoints.push_back(cv::Point(x, y));
        }
        
        // Fill the sector
        if (arcPoints.size() > 2) {
            cv::fillPoly(frame, arcPoints, progressColor);
        }
        
        // Draw border circle on top
        cv::circle(frame, center, roiRadius, cv::Scalar(255, 255, 255), 2);
    }
    
    // Draw percentage text in the center
    std::string progressText = std::to_string(static_cast<int>(progress * 100)) + "%";
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.6;
    int thickness = 2;
    
    // Get text size to center it
    cv::Size textSize = cv::getTextSize(progressText, fontFace, fontScale, thickness, nullptr);
    cv::Point textPos(center.x - textSize.width / 2, center.y + textSize.height / 2);
    
    // Draw text with background for better visibility
    cv::putText(frame, progressText, textPos, fontFace, fontScale, cv::Scalar(0, 0, 0), thickness + 2);
    cv::putText(frame, progressText, textPos, fontFace, fontScale, cv::Scalar(255, 255, 255), thickness);
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

// CardHandler implementation
CardHandler::CardHandler(const std::vector<cv::Rect>& cardAreas, const cv::Rect& playZone) {
    if (cardAreas.size() != 3) {
        throw std::invalid_argument("Player card areas must be exactly 3");
    }

    playerCardAreas = cardAreas;
    playArea = playZone;
}

std::vector<int> CardHandler::detectPlayerCards() {
    if (currentFrame.empty()) {
        throw std::runtime_error("Frame not set");
    }

    std::vector<int> cardIds;

    for (const auto& area : playerCardAreas) {
        cv::Mat cardRegion = currentFrame(area);
        ClassificationResult result = NetworkManager::classify(cardRegion);
        
        if (result.classId > 0) {
            cardIds.push_back(result.classId);
        }
    }

    return cardIds;
}

std::pair<int, std::vector<int>> CardHandler::detectPlayedCard() {
    if (currentFrame.empty()) {
        throw std::runtime_error("Frame not set");
    }

    // Detect current player cards
    std::vector<int> currentCards = detectPlayerCards();

    // Detect card in play area
    cv::Mat playRegion = currentFrame(playArea);
    ClassificationResult result = NetworkManager::classify(playRegion);

    if (result.classId <= 0) {
        throw std::runtime_error("No card detected in play area");
    }

    int playedCardId = result.classId;

    // Remove played card from player's hand
    std::vector<int> remainingCards;
    bool cardFound = false;

    for (int cardId : currentCards) {
        if (!cardFound && (cardId == playedCardId)) {
            cardFound = true; // Skip first matching card
        }
        else {
            remainingCards.push_back(cardId);
        }
    }

    return { playedCardId, remainingCards };
}
