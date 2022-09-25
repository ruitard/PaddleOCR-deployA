#include <filesystem>
#include <fstream>

#include "iniparser.hpp"
#include "paddle_infer.hpp"

namespace fs = std::filesystem;

static void print_result(const std::vector<PaddleOCR::OCRPredictResult> &ocr_result) {
    for (auto i = 0; i < ocr_result.size(); i++) { // NOLINT
        std::cout << i << ". ";
        const PaddleOCR::Box &box = ocr_result[i].box;
        if (!box.empty()) {
            std::cout << "det boxes: [";
            for (int n = 0; n < box.size(); n++) {
                std::cout << '[' << box[n][0] << ',' << box[n][1] << "]";
                if (n != box.size() - 1) {
                    std::cout << ',';
                }
            }
            std::cout << "] ";
        }
        // rec
        if (ocr_result[i].score != -1.0) {
            std::cout << "rec text: " << ocr_result[i].text << " rec score: " << ocr_result[i].score
                      << " ";
        }
        std::cout << std::endl;
    }
}

static void scan_file(const PaddleOCR::PaddleOCR &ocr, const fs::path &img_path) {
    auto results = ocr.ocr(img_path);
    print_result(results);
}

static void scan_target(const PaddleOCR::PaddleOCR &ocr, const fs::path &target_path) {
    if (fs::is_regular_file(target_path)) {
        scan_file(ocr, target_path);
    }
    if (!fs::is_directory(target_path)) {
        return;
    }
    for (const auto &entry : fs::recursive_directory_iterator(target_path)) {
        if (fs::is_regular_file(entry)) {
            scan_file(ocr, entry.path());
        }
    }
}

int main(int argc, const char *argv[]) {
    PaddleOCR::PaddleConfig config;
    if (std::ifstream ifs{"config.ini"}; ifs.is_open()) {
        inipp::Ini ini;
        if (ini.parse(ifs)) {
            config.det_model_dir = ini.must_get<std::string>("base", "det_model_dir");
            config.cls_model_dir = ini.must_get<std::string>("base", "cls_model_dir");
            config.rec_model_dir = ini.must_get<std::string>("base", "rec_model_dir");
            config.rec_char_dict_path = ini.must_get<std::string>("base", "rec_char_dict_path");
        }
    }
    const PaddleOCR::PaddleOCR ocr(config);
    for (int i = 1; i < argc; i++) {
        const fs::path target_path{argv[i]}; // NOLINT
        scan_target(ocr, target_path);
    }
    return 0;
}