#include "CardHandler.h"

CardHandler::CardHandler(const std::vector<cv::Rect>& cardAreas, const cv::Rect& playZone, CardClassifier* cls) {
    // Oyuncu kart alanlarını kontrol et (tam olarak 3 alan olmalı)
    if (cardAreas.size() != 3) {
        throw std::invalid_argument("Oyuncu kart alanları tam olarak 3 tane olmalıdır");
    }
    
    playerCardAreas = cardAreas;
    playArea = playZone;
    classifier = cls;
}

void CardHandler::setFrame(const cv::Mat& frame) {
    currentFrame = frame.clone();
}

std::vector<int> CardHandler::detectPlayerCards() {
    if (currentFrame.empty()) {
        throw std::runtime_error("Frame ayarlanmamış");
    }
    
    std::vector<int> cardIds;
    
    // Her bir kart alanı için detector kullanarak kartları tespit et
    for (const auto& area : playerCardAreas) {
        // İlgili alanı kes
        cv::Mat cardRegion = currentFrame(area);
        
        // Classifier kullanarak kart ID'sini al
        int cardId = classifier->classifyCard(cardRegion);
        
        // Tespit edilen kart ID'sini listeye ekle (sadece geçerli ID'ler)
        if (cardId > 0) {
            cardIds.push_back(cardId);
        }
        
    }
    
    return cardIds;
}

std::pair<int, std::vector<int>> CardHandler::detectPlayedCard() {
    if (currentFrame.empty()) {
        throw std::runtime_error("Frame ayarlanmamış");
    }
    
    // Önce oyuncunun mevcut kartlarını tespit et
    std::vector<int> currentCards = detectPlayerCards();
    
    // Oyun alanındaki kartı tespit et
    cv::Mat playRegion = currentFrame(playArea);
    int playedCardId = classifier->classifyCard(playRegion);
    
    // Oynanan kart yoksa hata fırlat
    if (playedCardId <= 0) {
        throw std::runtime_error("Oynanan kart tespit edilemedi");
    }
    
    // Oynanan kartı oyuncunun elinden çıkar
    std::vector<int> remainingCards;
    bool cardFound = false;
    
    for (int cardId : currentCards) {
        if (!cardFound && (cardId == playedCardId)) {
            cardFound = true; // İlk eşleşen kartı atla
        } else {
            remainingCards.push_back(cardId);
        }
    }
    
    return {playedCardId, remainingCards};
}
