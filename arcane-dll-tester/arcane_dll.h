#pragma once

#ifdef ARCANE_DLL_EXPORTS
#define ARCANE_DLL_API __declspec(dllexport)
#else
#define ARCANE_DLL_API __declspec(dllimport)
#endif

#include <opencv2/core.hpp>
#include <vector>
#include <chrono>


#define DICE_CLASS 0


struct DetectionResult {
    cv::Rect boundingBox;
    int classId;
    float confidence;
};

struct DetectionResultArray {
    DetectionResult* results;
    int size;
};

struct ClassificationResult {
    int classId;
    float confidence;
};


class CalibrationChecker {
private:
    cv::Mat currentFrame;

    // Kontrol edilecek dairelerin beklenen merkezleri ve yarýçaplarý
    std::vector<cv::Point> expectedCenters;
    std::vector<int> expectedRadii;

    // HSV renk aralýðý (yeþil için)
    cv::Scalar lowerGreen;
    cv::Scalar upperGreen;

    // Minimum daire algýlama boyutu
    int minRadius;

    // Tolerans deðerleri (piksel cinsinden)
    int centerTolerance;
    int radiusTolerance;

public:
    // Constructor
    CalibrationChecker(const std::vector<cv::Point>& centers, const std::vector<int>& radii);

    // Kareyi ayarla
    void setFrame(const cv::Mat& frame);

    // Kalibrasyon kontrolü yap
    bool checkCalibration();

    // HSV renk aralýðýný ayarla (opsiyonel)
    void setGreenRange(const cv::Scalar& lower, const cv::Scalar& upper);

    // Tolerans deðerlerini ayarla (opsiyonel)
    void setTolerances(int centerTol, int radiusTol);
};


class PlayerReady {
public:
    PlayerReady(int roiRadius, cv::Point center, int secondsToWait);

    void setFrame(const cv::Mat& newFrame);
    bool checkReady(); // her karede çaðrýlýr, ama zaman içinde karar verir

private:
    int roiRadius;
    cv::Point center;
    cv::Mat frame;
    int secondsToWait = 3;

    bool hasSeenGreen = false;
    std::chrono::time_point<std::chrono::high_resolution_clock> startTime;
    bool timerStarted = false;

    cv::Scalar lowerGreen = cv::Scalar(35, 50, 50);
    cv::Scalar upperGreen = cv::Scalar(85, 255, 255);

    bool isGreenCovered();
};

class CardHandler {
private:
    cv::Mat currentFrame;
    std::vector<cv::Rect> playerCardAreas; // 3 oyuncu kart alaný
    cv::Rect playArea;                     // Kartýn oynandýðý alan

    // Classifier sýnýfý için pointer
    ClassificationResult classifier;

public:
    // Constructor: 4 alan koordinatlarýný ve classifier objesini al
    CardHandler(const std::vector<cv::Rect>& cardAreas, const cv::Rect& playZone, ClassificationResult cls);

    // Ýþlenecek kareyi güncelle
    void setFrame(const cv::Mat& frame);

    // Oyuncunun elindeki kartlarý tespit et
    std::vector<int> detectPlayerCards();

    // Oyuncunun kullandýðý kartý tespit et ve kalan kartlarý döndür
    std::pair<int, std::vector<int>> detectPlayedCard();
};






extern "C" {
    ARCANE_DLL_API bool InitializeNetworks(const char* detectionModelPath, const char* classificationModelPath, bool useCuda);
    ARCANE_DLL_API DetectionResultArray Detect(const cv::Mat& frame); // Updated function signature
    ARCANE_DLL_API ClassificationResult Classify(const cv::Mat& frame);
    ARCANE_DLL_API void Cleanup();
    ARCANE_DLL_API void setCalibrationProperties(cv::Point centers[2], const int radiuses[2]);
    ARCANE_DLL_API bool checkCalibrationApi(const cv::Mat& frame);
    ARCANE_DLL_API void setReadyProperties(cv::Point center, const int radius, int secondsToWait);
    ARCANE_DLL_API bool checkPlayerReady(const cv::Mat& frame);
}