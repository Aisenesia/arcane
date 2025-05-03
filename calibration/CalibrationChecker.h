#ifndef CALIBRATION_CHECKER_H
#define CALIBRATION_CHECKER_H

#include <opencv2/opencv.hpp>
#include <vector>

class CalibrationChecker {
private:
    cv::Mat currentFrame;
    
    // Kontrol edilecek dairelerin beklenen merkezleri ve yarıçapları
    std::vector<cv::Point> expectedCenters;
    std::vector<int> expectedRadii;
    
    // HSV renk aralığı (yeşil için)
    cv::Scalar lowerGreen;
    cv::Scalar upperGreen;
    
    // Minimum daire algılama boyutu
    int minRadius;
    
    // Tolerans değerleri (piksel cinsinden)
    int centerTolerance;
    int radiusTolerance;

public:
    // Constructor
    CalibrationChecker(const std::vector<cv::Point>& centers, const std::vector<int>& radii);
    
    // Kareyi ayarla
    void setFrame(const cv::Mat& frame);
    
    // Kalibrasyon kontrolü yap
    bool checkCalibration();
    
    // HSV renk aralığını ayarla (opsiyonel)
    void setGreenRange(const cv::Scalar& lower, const cv::Scalar& upper);
    
    // Tolerans değerlerini ayarla (opsiyonel)
    void setTolerances(int centerTol, int radiusTol);
};

#endif // CALIBRATION_CHECKER_H