#include "CardClassifier.h"

CardClassifier::CardClassifier() {
    // Boş constructor
}

int CardClassifier::classifyCard(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("Boş görüntü verildi.");
    }

    // BGR'den HSV'ye çevir
    cv::Mat hsv;
    cv::cvtColor(image, hsv, cv::COLOR_BGR2HSV);

    // Renk aralıklarını tanımla
    cv::Mat redMask1, redMask2, greenMask, blueMask;

    // Kırmızı iki aralıkta bulunur (0-10 ve 160-180)
    cv::inRange(hsv, cv::Scalar(0, 70, 50), cv::Scalar(10, 255, 255), redMask1);
    cv::inRange(hsv, cv::Scalar(160, 70, 50), cv::Scalar(180, 255, 255), redMask2);
    cv::bitwise_or(redMask1, redMask2, redMask1);

    // Yeşil
    cv::inRange(hsv, cv::Scalar(35, 70, 50), cv::Scalar(85, 255, 255), greenMask);

    // Mavi
    cv::inRange(hsv, cv::Scalar(100, 70, 50), cv::Scalar(130, 255, 255), blueMask);

    // Her maskede kaç piksel var
    int redCount = cv::countNonZero(redMask1);
    int greenCount = cv::countNonZero(greenMask);
    int blueCount = cv::countNonZero(blueMask);

    // En çok hangi renk var?
    if (redCount > greenCount && redCount > blueCount)
        return 1; // Kırmızı
    else if (greenCount > redCount && greenCount > blueCount)
        return 2; // Yeşil
    else if (blueCount > redCount && blueCount > greenCount)
        return 3; // Mavi
    else
        return 0; // Tanımsız / eşitlik durumu
}
