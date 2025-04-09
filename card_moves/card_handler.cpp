#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

/*
g++ -std=c++17 -o card_handler card_handler.cpp `pkg-config --cflags --libs opencv4`
*/

using namespace cv;
using namespace std;

class CardHandler {
public:
    CardHandler(const vector<Rect>& rois) : rois(rois) {}

    // Kırmızı kartın belirtilen ROI'de olup olmadığını kontrol eder
    bool isRedCardPresent(const Mat& frame, const Rect& roi) {
        Mat roi_img = frame(roi);
        // Kırmızı renk aralığı (HSV renk uzayında)
        Scalar lower_red = Scalar(0, 100, 100);
        Scalar upper_red = Scalar(10, 255, 255);
        Scalar lower_red2 = Scalar(160, 100, 100);
        Scalar upper_red2 = Scalar(179, 255, 255);


        Mat hsv_img;
        cvtColor(roi_img, hsv_img, COLOR_BGR2HSV);

        Mat mask1, mask2, mask;
        inRange(hsv_img, lower_red, upper_red, mask1);
        inRange(hsv_img, lower_red2, upper_red2, mask2);
        bitwise_or(mask1, mask2, mask);


        // Kırmızı piksel sayısını kontrol et
        int red_pixel_count = countNonZero(mask);
        // Belirli bir eşik değerini aşarsa kırmızı kart var say
        return red_pixel_count > roi.area() * 0.1; // %10'u kırmızı ise kart var
    }

    // Kart hareketlerini kontrol eder ve değişimleri yazdırır
    void checkCardMovements(const Mat& frame) {
        // Önceki frame'deki kart konumlarını sakla
        static vector<bool> previous_card_positions(rois.size(), false);

        for (size_t i = 0; i < rois.size(); ++i) {
            bool current_card_present = isRedCardPresent(frame, rois[i]);
            if (current_card_present)
            {
                rectangle(frame, rois[i], Scalar(0, 0, 255), 2);
            }
            
            // Karşılıklı alan indeksini hesapla (varsayılan olarak i + 1, son alan için 0)
            size_t opposite_index = (i + 1) % rois.size();


            if (current_card_present && !previous_card_positions[i] && previous_card_positions[opposite_index]) {
                cout << "Kart " << opposite_index + 1 << ". alandan " << i + 1 << ". alana taşındı." << endl;
            }

            previous_card_positions[i] = current_card_present;
        }
    }

private:
    vector<Rect> rois;
};


int main() {
    // Dikdörtgen alanları tanımla (ROI'ler)
    vector<Rect> rois = {
        Rect(100, 100, 100, 140),  // Alan 1
        Rect(300, 100, 100, 140),  // Alan 2
        Rect(500, 100, 100, 140),  // Alan 3
        Rect(700, 100, 100, 140)   // Alan 4 (örnek, istediğiniz kadar ekleyebilirsiniz)
    };

    CardHandler cardHandler(rois);

    VideoCapture capture(0);

    while (true) {
        Mat frame;
        capture >> frame;

        if (frame.empty()) {
            break;
        }

        cardHandler.checkCardMovements(frame);

        // ROI'leri görselleştir (isteğe bağlı)
        for (const auto& roi : rois) {
            rectangle(frame, roi, Scalar(0, 255, 0), 2);
            putText(frame, "ROI", Point(roi.x, roi.y - 10), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 0), 2);
        }


        imshow("Kart Hareket Dedektoru", frame);

        if (waitKey(1) == 27) {
            break;
        }
    }

    return 0;
}