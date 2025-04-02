#include <algorithm>
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/cudaimgproc.hpp> // For CUDA-based preprocessing
#include <opencv2/cudawarping.hpp>

using namespace cv;
using namespace cv::dnn;
using namespace std;

// Function to convert class IDs
int classConverter(int classId) {
    map<int, int> class_mapping = {
        {0, 1}, {1, 2}, {12, 3}, {13, 4}, {14, 5}, {15, 6},
        {16, 7}, {17, 8}, {18, 9}, {19, 10},
        {2, 11}, {3, 12}, {4, 13}, {5, 14}, {6, 15},
        {7, 16}, {8, 17}, {9, 18}, {10, 19}, {11, 20}
    };

    if (class_mapping.find(classId) != class_mapping.end()) {
        return class_mapping[classId];
    } else {
        return -1;
    }
}

// Function to classify a cropped image
void classifyImage(Net& netClassification, const Mat& image) {
    Mat blob;
    blobFromImage(image, blob, 1.0 / 255.0, Size(224, 224), Scalar(), true, false);
    netClassification.setInput(blob);

    Mat output = netClassification.forward();

    Point classIdPoint;
    double confidence;
    minMaxLoc(output, 0, &confidence, 0, &classIdPoint);
    int classId = classConverter(classIdPoint.x);

    cout << "Classified as: " << classId << " with confidence: " << confidence << endl;
}

// Function to detect dice in a frame
void detectDice(Net& netDetection, Net& netClassification, Mat& frame) {
    // Preprocess the frame using CUDA
    cv::cuda::GpuMat gpuFrame;
    gpuFrame.upload(frame); // Upload frame to GPU

    // Resize the frame on the GPU
    cv::cuda::GpuMat resizedGpuFrame;
    cv::cuda::resize(gpuFrame, resizedGpuFrame, Size(640, 640));

    // Download the resized frame back to the CPU
    Mat resizedFrame;
    resizedGpuFrame.download(resizedFrame);

    // Create a blob from the resized frame
    Mat blob;
    blobFromImage(resizedFrame, blob, 1.0 / 255.0, Size(640, 640), Scalar(), true, false);

    // Set the blob as input to the detection network
    netDetection.setInput(blob);

    // Get detection output
    Mat output = netDetection.forward();

    if (output.dims == 3 && output.size[1] == 5) {
        vector<Rect> boxes;
        vector<float> confidences;

        int numDetections = output.size[2];
        for (int i = 0; i < numDetections; i++) {
            float confidence = output.ptr<float>(0)[4 * numDetections + i];
            if (confidence > 0.5) {
                float cx = output.ptr<float>(0)[0 * numDetections + i];
                float cy = output.ptr<float>(0)[1 * numDetections + i];
                float w = output.ptr<float>(0)[2 * numDetections + i];
                float h = output.ptr<float>(0)[3 * numDetections + i];

                int left = static_cast<int>((cx - w / 2) * frame.cols / 640);
                int top = static_cast<int>((cy - h / 2) * frame.rows / 640);
                int width = static_cast<int>(w * frame.cols / 640);
                int height = static_cast<int>(h * frame.rows / 640);

                left = max(0, min(left, frame.cols - 1));
                top = max(0, min(top, frame.rows - 1));
                width = min(width, frame.cols - left);
                height = min(height, frame.rows - top);

                if (width > 0 && height > 0) {
                    Rect box(left, top, width, height);
                    boxes.push_back(box);
                    confidences.push_back(confidence);
                }
            }
        }

        // Apply non-maximum suppression
        vector<int> indices;
        if (!boxes.empty()) {
            NMSBoxes(boxes, confidences, 0.5, 0.4, indices);
        }

        // Process surviving detections
        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            Rect box = boxes[idx];

            Mat cropped = frame(box).clone();
            if (!cropped.empty()) {
                classifyImage(netClassification, cropped);
                rectangle(frame, box, Scalar(0, 255, 0), 2);
            }
        }
    }
    imshow("YOLO Dice Detection", frame);
}

// Main function
int main(int argc, char** argv) {
    string detectionModelPath = "yolov8m_detection.onnx";
    string classificationModelPath = "yolov8s_cls.onnx";

    // Load detection and classification models
    Net netDetection = readNetFromONNX(detectionModelPath);
    netDetection.setPreferableBackend(DNN_BACKEND_CUDA);
    netDetection.setPreferableTarget(DNN_TARGET_CUDA_FP16); // Use FP16 for better performance

    Net netClassification = readNetFromONNX(classificationModelPath);
    netClassification.setPreferableBackend(DNN_BACKEND_CUDA);
    netClassification.setPreferableTarget(DNN_TARGET_CUDA_FP16);

    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cout << "Error opening camera" << endl;
        return -1;
    }

    Mat frame;
    while (cap.read(frame)) {
        detectDice(netDetection, netClassification, frame);
        if (waitKey(1) == 27) break;  // Press ESC to exit
    }

    cap.release();
    destroyAllWindows();
    return 0;
}

