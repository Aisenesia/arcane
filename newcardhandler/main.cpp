#include "CardHandler.h"
#include "CardClassifier.h"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <unistd.h>

int main() {
    // Kamera aç
    cv::VideoCapture cap(0);
    if (!cap.isOpened()) {
        std::cerr << "Kamera açılamadı!" << std::endl;
        return -1;
    }

    // Kart alanlarını manuel olarak tanımla (örnek değerler, çözünürlük 640x480 varsayılmış)
    std::vector<cv::Rect> cardAreas = {
        cv::Rect(50, 380, 100, 80),   // Sol
        cv::Rect(270, 380, 100, 80),  // Orta
        cv::Rect(490, 380, 100, 80)   // Sağ
    };

    cv::Rect playArea(270, 50, 100, 80);  // Oyun alanı

    // Classifier ve handler nesneleri oluştur
    CardClassifier classifier;
    CardHandler handler(cardAreas, playArea, &classifier);

    while (true) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) break;

        handler.setFrame(frame);

        // Oyuncu kartlarını algıla
        std::vector<int> playerCards = handler.detectPlayerCards();

        std::cout << "Oyuncu kartları: ";
        for (int id : playerCards) std::cout << id << " ";
        std::cout << std::endl;

        // Oynanan kartı algıla
        try {
            auto [playedCard, remaining] = handler.detectPlayedCard();
            std::cout << "Oynanan kart: " << playedCard << std::endl;
            std::cout << "Kalan kartlar: ";
            for (int id : remaining) std::cout << id << " ";
            std::cout << std::endl;
        } catch (const std::exception& e) {
            std::cout << "Oynanan kart tespit edilemedi: " << e.what() << std::endl;
        }

        // Görsel gösterim (alanları çiz)
        for (const auto& r : cardAreas) {
            cv::rectangle(frame, r, cv::Scalar(0, 255, 0), 2);
        }
        cv::rectangle(frame, playArea, cv::Scalar(0, 0, 255), 2);

        cv::imshow("Kart Algılama", frame);
        if (cv::waitKey(30) == 27) break;
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}
