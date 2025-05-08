#include "CalibrationChecker.h"
#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Beklenen merkezler ve yarıçaplar (örnek değerler)
    std::vector<cv::Point> centers = {cv::Point(150, 150), cv::Point(1130, 570)};
    std::vector<int> radii = {60, 60};

    CalibrationChecker checker(centers, radii);

    // Kamera aç (varsayılan olarak 0. cihaz)
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Kamera açılamadı!" << std::endl;
        return -1;
    }

    while (true) {
        cv::Mat frame;
        cap >> frame;  // Kameradan görüntü al

        if (frame.empty()) {
            std::cerr << "Boş kare alındı!" << std::endl;
            break;
        }

        //printf("Görüntü boyutu: %d x %d\n", frame.cols, frame.rows);

        // Görüntüyü yata yçevir
        //cv::flip(frame, frame, 1);

        // kontro ledilen dairelerin merkezlerini ve yarıçaplarını çiz
        for (size_t i = 0; i < centers.size(); ++i) {
            cv::circle(frame, centers[i], radii[i], cv::Scalar(255, 0, 0), 2);
        }


        checker.setFrame(frame);
        bool result = checker.checkCalibration();

        // Sonucu görüntüye yaz
        std::string status = result ? "KALIBRASYON TAM" : "KALIBRASYON HATALI";
        cv::putText(frame, status, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 1,
                    result ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255), 2);

        // Görüntüyü göster
        cv::imshow("Kalibrasyon Kontrol", frame);

        // ESC tuşuna basılırsa çık
        char key = (char)cv::waitKey(50);
        if (key == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
