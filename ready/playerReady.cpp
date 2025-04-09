#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>

using namespace cv;
using namespace std;
using namespace std::chrono;

/*
g++ -std=c++17 -o playerReady playerReady.cpp `pkg-config --cflags --libs opencv4`
*/

class RedDetector {
public:
    RedDetector(int cameraIndex = 0, int roiRadius = 100) : 
        cap(cameraIndex), roiRadius(roiRadius), hazir(false) {

        if (!cap.isOpened()) {
            cerr << "Kamera açılamadı!" << endl;
            exit(-1); // Hata durumunda programdan çık
        }

        // Dairenin merkezi, frame boyutuna göre başlatılır
        center = Point(cap.get(CAP_PROP_FRAME_WIDTH) / 2, cap.get(CAP_PROP_FRAME_HEIGHT) / 2);
    }

    void run() {
        while (true) {
            Mat frame;
            cap >> frame;

            if (frame.empty()) {
                cerr << "Frame alınamadı!" << endl;
                break;
            }

            if (isRedCovered(frame)) {
                auto currentTime = high_resolution_clock::now();
                auto duration = duration_cast<seconds>(currentTime - startTime);

                // Kırmızı kapalı kalma süresini göster
                putText(frame, "Kirmizi Kapali", Point(50, 100), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);
                // Süreyi göster
                putText(frame, to_string(duration.count()) + " sn", Point(50, 150), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 0, 255), 2);


                if (duration.count() >= 10 && !hazir) {
                    hazir = true;
                    cout << "Oyuncu Hazır!" << endl;
                    putText(frame, "Oyuncu Hazır!", Point(50, 50), FONT_HERSHEY_SIMPLEX, 1, Scalar(0, 255, 0), 2);
                }
            } else {
                startTime = high_resolution_clock::now();
                hazir = false;
            }

            // ROI'yi çiz
            circle(frame, center, roiRadius, Scalar(0, 255, 0), 2);
            imshow("Frame", frame);

            if (waitKey(1) == 27) {
                break;
            }
        }

        cap.release();
        destroyAllWindows();
    }

private:
    VideoCapture cap;
    Point center;
    int roiRadius;
    bool hazir;
    high_resolution_clock::time_point startTime;

    Scalar lowerRed = Scalar(0, 100, 100);
    Scalar upperRed = Scalar(10, 255, 255);
    Scalar lowerRed2 = Scalar(160, 100, 100);
    Scalar upperRed2 = Scalar(179, 255, 255);


    bool isRedCovered(Mat& frame) {
        Mat mask = Mat::zeros(frame.size(), CV_8UC1);
        circle(mask, center, roiRadius, Scalar(255), FILLED);
    
        Mat hsvFrame;
        cvtColor(frame, hsvFrame, COLOR_BGR2HSV);
    
        Mat redMask, redMask2, combinedRedMask;
        inRange(hsvFrame, lowerRed, upperRed, redMask);
        inRange(hsvFrame, lowerRed2, upperRed2, redMask2);
        bitwise_or(redMask, redMask2, combinedRedMask);
    
        Mat roiRedMask;
        bitwise_and(combinedRedMask, mask, roiRedMask);
    
        // ROI'deki toplam piksel sayısı
        int totalRoiPixels = countNonZero(mask);
    
        // ROI'de kırmızı olmayan piksel sayısı
        int nonRedPixels = countNonZero(roiRedMask);
    
    
        // Kapsama oranını hesapla
        double coverageRatio = (double)nonRedPixels / totalRoiPixels;
    
        // Kapsama oranı %60'dan büyükse true döndür
        return coverageRatio <= 0.4; // %60 kapalı ise %40'ı görünür demektir.
    }
};

int main() {
    RedDetector detector; // Varsayılan kamera ve yarıçap
    //RedDetector detector(1, 50); // Farklı kamera ve yarıçap için

    detector.run();

    return 0;
}