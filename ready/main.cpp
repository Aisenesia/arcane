#include "PlayerReady.h"
#include <opencv2/opencv.hpp>
#include <iostream>
int main() {
    PlayerReady playerReady(50, cv::Point(100, 100));

    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Kamera açılamadı!" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;

        if (frame.empty()) {
            std::cerr << "Boş kare alındı!" << std::endl;
            break;
        }

        playerReady.setFrame(frame);

        bool ready = playerReady.checkReady(3);
        if (ready) {
            std::cout << "Player is ready!" << std::endl;
        }

        cv::circle(frame, cv::Point(100, 100), 50, cv::Scalar(0, 255, 0), 2);
        cv::imshow("Kamera Görüntüsü", frame);

        if (cv::waitKey(30) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
