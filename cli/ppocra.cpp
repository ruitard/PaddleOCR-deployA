#include <iostream>
#include <filesystem>
#include "paddle_infer.hpp"

namespace fs = std::filesystem;

static void print_result(const std::vector<PaddleOCR::OCRPredictResult> &ocr_result) {
    for (int i = 0; i < ocr_result.size(); i++) {
        std::cout << i << "\t";
        // det
        std::vector<std::vector<int>> boxes = ocr_result[i].box;
        if (boxes.size() > 0) {
            std::cout << "det boxes: [";
            for (int n = 0; n < boxes.size(); n++) {
                std::cout << '[' << boxes[n][0] << ',' << boxes[n][1] << "]";
                if (n != boxes.size() - 1) {
                    std::cout << ',';
                }
            }
            std::cout << "] ";
        }
        // rec
        if (ocr_result[i].score != -1.0) {
            std::cout << "rec text: " << ocr_result[i].text << " rec score: " << ocr_result[i].score << " ";
        }

        // cls
        if (ocr_result[i].cls_label != -1) {
            std::cout << "cls label: " << ocr_result[i].cls_label << " cls score: " << ocr_result[i].cls_score;
        }
        std::cout << std::endl;
    }
}

int main(int argc, char *argv[]) {
    fs::path img_path{argv[1]};
    PaddleOCR::PaddleOCR ocr;
    auto results = ocr.ocr(img_path);
    print_result(results);
    return 0;
}