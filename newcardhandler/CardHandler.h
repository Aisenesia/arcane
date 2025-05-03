#ifndef CARD_HANDLER_H
#define CARD_HANDLER_H

#include <opencv2/opencv.hpp>
#include <vector>
#include "CardClassifier.h"

class CardHandler {
private:
    cv::Mat currentFrame;
    std::vector<cv::Rect> playerCardAreas; // 3 oyuncu kart alanı
    cv::Rect playArea;                     // Kartın oynandığı alan
    
    // Classifier sınıfı için pointer
    CardClassifier* classifier;

public:
    // Constructor: 4 alan koordinatlarını ve classifier objesini al
    CardHandler(const std::vector<cv::Rect>& cardAreas, const cv::Rect& playZone, CardClassifier* cls);
    
    // İşlenecek kareyi güncelle
    void setFrame(const cv::Mat& frame);
    
    // Oyuncunun elindeki kartları tespit et
    std::vector<int> detectPlayerCards();
    
    // Oyuncunun kullandığı kartı tespit et ve kalan kartları döndür
    std::pair<int, std::vector<int>> detectPlayedCard();
};

#endif // CARD_HANDLER_H