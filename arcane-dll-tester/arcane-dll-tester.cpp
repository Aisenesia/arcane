#include "arcane_dll.h"
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudawarping.hpp>
#include <windows.h> // For GetSystemMetrics
#include <vector>
#include <filesystem> // For checking file existence
#include <opencv2/dnn.hpp> // For NMSBoxes

#define NOMINMAX

// Window management constants
const char* WINDOW_NAME = "Camera View";
const int WINDOW_PADDING = 50;  // Padding from screen edges
const double DEFAULT_SCALE_FACTOR = 0.8;  // Default scale factor for video

// Struct to store display settings
struct DisplaySettings {
    int screenWidth;
    int screenHeight;
    double maxWindowWidth;
    double maxWindowHeight;
};

// Create and position a window properly
void setupWindow() {
    cv::namedWindow(WINDOW_NAME, cv::WINDOW_NORMAL);
    cv::resizeWindow(WINDOW_NAME, 1600, 720);
}

void applyNonMaximumSuppression(const DetectionResultArray& arr, std::vector<cv::Rect>& filteredBoxes, std::vector<float>& confidences, float nmsThreshold = 0.4) {
    std::vector<int> indices;
    for (int i = 0; i < arr.size; i++) {
        filteredBoxes.push_back(arr.results[i].boundingBox);
        confidences.push_back(arr.results[i].confidence);
    }

    // Apply Non-Maximum Suppression
    cv::dnn::NMSBoxes(filteredBoxes, confidences, 0.5, nmsThreshold, indices);

    // Filter out boxes based on NMS results
    std::vector<cv::Rect> nmsFilteredBoxes;
    for (int idx : indices) {
        nmsFilteredBoxes.push_back(filteredBoxes[idx]);
    }

    filteredBoxes = nmsFilteredBoxes;
}

void processReadyCommand(cv::VideoCapture& cap) {
	setupWindow();

    cv::Point center(100, 100);
    int radius = 50;
    int secondsToWait = 3;

    setReadyProperties(center, radius, secondsToWait);

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Error: Empty frame received! Exiting 'ready' mode." << std::endl;
            break;
        }

        bool ready = checkPlayerReady(frame);
        if (ready) {
            std::cout << "Player is ready!" << std::endl;
        }

        cv::circle(frame, center, radius, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Camera View", frame);

        cv::resizeWindow(WINDOW_NAME, frame.cols, frame.rows);


        if (cv::waitKey(30) == 27) break; // Exit on 'ESC' key
    }





    cv::destroyWindow("Camera View");
}

void processDiceCommand(cv::VideoCapture& cap) {
    const char* detectionModelPath = "dtc.onnx";
    const char* classificationModelPath = "cls.onnx";

    if (!InitializeNetworks(detectionModelPath, classificationModelPath, true)) {
        std::cerr << "Failed to initialize networks. Please check the model files and paths." << std::endl;
        return;
    }

    setupWindow();

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Empty frame received!" << std::endl;
            break;
        }

        DetectionResultArray arr = Detect(frame);
        std::vector<cv::Rect> filteredBoxes;
        std::vector<float> confidences;
        applyNonMaximumSuppression(arr, filteredBoxes, confidences);

        for (size_t i = 0; i < filteredBoxes.size(); ++i) {
            cv::rectangle(frame, filteredBoxes[i], cv::Scalar(0, 255, 0), 2);
            std::string label = "Class: " + std::to_string(arr.results[i].classId);
            cv::putText(frame, label, cv::Point(filteredBoxes[i].x, filteredBoxes[i].y - 10),
                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 0), 1);
        }

        cv::imshow(WINDOW_NAME, frame);

        // Ensure window fits properly on screen
        cv::resizeWindow(WINDOW_NAME, frame.cols, frame.rows);

        if (cv::waitKey(30) == 27) break; // Exit on 'ESC' key
    }
}

void processCalibrateCommand(cv::VideoCapture& cap) {
    std::cout << "Calibration mode activated." << std::endl;
    cv::Point centers[] = { cv::Point(150, 150), cv::Point(1130, 570) };
    int radii[2] = { 60, 60 };
    setCalibrationProperties(centers, radii);

    setupWindow();

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Empty frame received!" << std::endl;
            break;
        }

        for (size_t i = 0; i < 2; ++i) {
            cv::circle(frame, centers[i], radii[i], cv::Scalar(255, 0, 0), 2);
        }

        bool result = checkCalibrationApi(frame);
        std::string status = result ? "KALIBRASYON TAM" : "KALIBRASYON HATALI";
        cv::putText(frame, status, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
            result ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        cv::imshow(WINDOW_NAME, frame);

        // Ensure window fits properly on screen
        cv::resizeWindow(WINDOW_NAME, frame.cols, frame.rows);

        if (cv::waitKey(30) == 27) break; // Exit on 'ESC' key
    }
}


int main(int argc, char* argv[]) {
    if (argc > 1) {
        std::string command = argv[1];

        // List the available cameras
        std::cout << "Available cameras:" << std::endl;
        for (int i = 0; i < 10; ++i) {
            cv::VideoCapture testCap(i);
            if (testCap.isOpened()) {
                std::cout << "Camera " << i << " is available." << std::endl;
                testCap.release();
            }
        }
        int cameraIndex = 0;
        if(argc == 3) {
            cameraIndex = std::stoi(argv[2]);
            std::cout << "Using camera index: " << cameraIndex << std::endl;
        } else {
            std::cout << "Using default camera index: 0" << std::endl;
        }

        cv::VideoCapture cap (cameraIndex);
        if (!cap.isOpened()) {
            std::cerr << "Camera could not be opened!" << std::endl;
            return -1;
        }

        // Give camera time to initialize
        std::cout << "Initializing camera..." << std::endl;
        cv::Mat dummy;
        for (int i = 0; i < 10; i++) {
            cap >> dummy; // Discard initial frames that might be invalid
            cv::waitKey(100);
        }

        if (command == "--ready") {
            processReadyCommand(cap);
        }
        else if (command == "--dice") {
            processDiceCommand(cap);
        }
        else if (command == "--calibrate") {
            processCalibrateCommand(cap);
        }
        else {
            std::cerr << "Invalid command: " << command << std::endl;
        }

        cap.release();
    }
    /*
    // Test detection on an image
    const char* testImagePath = "sample.png";
    cv::Mat testImage = cv::imread(testImagePath);

    // Create a named window for the test image
    setupWindow();
    DisplaySettings settings = getDisplaySettings();

    DetectionResultArray arr = Detect(testImage);
    int size = arr.size;
    std::vector<DetectionResult> dices;

    for (int i = 0; i < size; i++) {
        std::cout << "Detection " << i << ": "
            << "Class ID: " << arr.results[i].classId << ", "
            << "Confidence: " << arr.results[i].confidence << ", "
            << "Bounding Box: (" << arr.results[i].boundingBox.x << ", "
            << arr.results[i].boundingBox.y << ", "
            << arr.results[i].boundingBox.width << ", "
            << arr.results[i].boundingBox.height << ")"
            << std::endl;
        if (arr.results[i].classId == DICE_CLASS) {
            dices.push_back(arr.results[i]);
        }

        // Draw bounding boxes on the image
        cv::rectangle(testImage, arr.results[i].boundingBox, cv::Scalar(0, 255, 0), 2);
    }

    // Display the test image with detections
    cv::Mat scaledTestImage = scaleToFitScreen(testImage, settings);
    cv::imshow(WINDOW_NAME, scaledTestImage);
    cv::resizeWindow(WINDOW_NAME, scaledTestImage.cols, scaledTestImage.rows);
    cv::waitKey(0);  // Wait for key press before continuing

    // Test classification on an image
    for (const auto& dice : dices) {
        cv::Mat diceImage = testImage(dice.boundingBox);
        ClassificationResult classificationResult = Classify(diceImage);
        std::cout << "Classification: "
            << "Class ID: " << classificationResult.classId << ", "
            << "Confidence: " << classificationResult.confidence
            << std::endl;
    }

    // Release GPU resources before exiting
    */
    Cleanup();
    cv::destroyAllWindows();

    return 0;
}