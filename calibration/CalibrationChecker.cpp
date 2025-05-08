#include "CalibrationChecker.h"
#include <iostream>

CalibrationChecker::CalibrationChecker(const std::vector<cv::Point>& centers, const std::vector<int>& radii) {
    // Tam olarak 2 merkez ve 2 yarıçap olmalı
    if (centers.size() != 2 || radii.size() != 2) {
        throw std::invalid_argument("Tam olarak 2 merkez ve 2 yarıçap olmalıdır");
    }
    
    expectedCenters = centers;
    expectedRadii = radii;
    
    // Varsayılan yeşil HSV aralığı
    lowerGreen = cv::Scalar(35, 100, 100);  // Açık yeşil
    upperGreen = cv::Scalar(85, 255, 255);  // Koyu yeşil
    
    // Varsayılan değerler
    minRadius = 10;  // Minimum 10 piksel yarıçaplı daireler algılanacak
    centerTolerance = 20;  // Merkez ±20 piksel olabilir
    radiusTolerance = 10;  // Yarıçap ±10 piksel olabilir
}

void CalibrationChecker::setFrame(const cv::Mat& frame) {
    currentFrame = frame.clone();
}

bool CalibrationChecker::checkCalibration() {
    if (currentFrame.empty()) {
        std::cerr << "Frame ayarlanmamış!" << std::endl;
        return false;
    }
    
    // Görüntüyü HSV'ye dönüştür
    cv::Mat hsvFrame;
    cv::cvtColor(currentFrame, hsvFrame, cv::COLOR_BGR2HSV);
    
    // Yeşil renk aralığını maskele
    cv::Mat mask;
    cv::inRange(hsvFrame, lowerGreen, upperGreen, mask);
    
    // // Gürültüyü azaltmak için morfolojik işlemler
    // cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(5, 5));
    // cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
    // cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
    
    // Daireleri algıla (HoughCircles algoritması kullanarak)
    std::vector<cv::Vec3f> circles;
    cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, 1, 
                   mask.rows/8,  // Minimum iki daire arasındaki mesafe
                   100, 30,      // Canny ve merkez eşik değerleri
                   minRadius, minRadius * 100);  // Min ve max yarıçap
    
    // Yeterli daire bulunamadıysa
    if (circles.size() < 2) {
        //std::cout << "Yeterli yeşil daire bulunamadı. Algılanan daire sayısı: " << circles.size() << std::endl;
        return false;
    }
    
    // En büyük iki daireyi seç
    std::sort(circles.begin(), circles.end(), 
              [](const cv::Vec3f& a, const cv::Vec3f& b) { return a[2] > b[2]; });
    
    // İlk iki daireyi al
    std::vector<cv::Vec3f> topCircles(circles.begin(), circles.begin() + std::min(2, static_cast<int>(circles.size())));
    
    bool allAligned = true;
    
    // Her beklenen merkezi kontrol et
    for (size_t i = 0; i < expectedCenters.size(); ++i) {
        // En yakın daireyi bul
        float minDist = std::numeric_limits<float>::max();
        int bestMatch = -1;
        
        for (size_t j = 0; j < topCircles.size(); ++j) {
            float dx = expectedCenters[i].x - topCircles[j][0];
            float dy = expectedCenters[i].y - topCircles[j][1];
            float dist = std::sqrt(dx*dx + dy*dy);
            
            if (dist < minDist) {
                minDist = dist;
                bestMatch = j;
            }
        }
        
        // En yakın daire yeteri kadar yakın mı?
        if (bestMatch >= 0) {
            // Merkez mesafesini kontrol et
            if (minDist > centerTolerance) {
                std::cout << "Daire #" << i << " merkezi tolerans dışında. Uzaklık: " << minDist << std::endl;
                allAligned = false;
            }
            
            // Yarıçapı kontrol et
            float radiusDiff = std::abs(expectedRadii[i] - topCircles[bestMatch][2]);
            if (radiusDiff > radiusTolerance) {
                std::cout << "Daire #" << i << " yarıçapı tolerans dışında. Fark: " << radiusDiff << std::endl;
                allAligned = false;
            }
        } else {
            std::cout << "Daire #" << i << " için eşleşme bulunamadı!" << std::endl;
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
