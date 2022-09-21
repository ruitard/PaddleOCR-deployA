#include <vector>
#include <string>
#include <memory>
#include <filesystem>

namespace PaddleOCR {

namespace fs = std::filesystem;

struct OCRPredictResult {
    std::vector<std::vector<int>> box;
    std::string text;
    float score = -1.0;
    float cls_score;
    int cls_label = -1;
};

class PaddleOCR {
public:
    PaddleOCR();
    PaddleOCR(const PaddleOCR &) = delete;
    PaddleOCR(PaddleOCR &&) = delete;
    PaddleOCR &operator=(const PaddleOCR &) = delete;
    PaddleOCR &operator=(PaddleOCR &&) = delete;

    std::vector<OCRPredictResult> ocr(const fs::path &image_path);
};

} // namespace PaddleOCR