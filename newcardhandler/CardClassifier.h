#ifndef CARD_CLASSIFIER_H
#define CARD_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include <string>

class CardClassifier {
public:
    // Constructor
    CardClassifier();
    
    // Kart sınıflandırma metodu - şimdilik sadece 1 döndürecek
    int classifyCard(const cv::Mat& image);
};

#endif // CARD_CLASSIFIER_H