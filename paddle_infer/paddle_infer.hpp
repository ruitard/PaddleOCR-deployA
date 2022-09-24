#include <vector>
#include <string>
#include <memory>
#include <filesystem>
#include <array>

namespace PaddleOCR {

namespace fs = std::filesystem;

using Point = std::array<int, 2>;
using Box = std::array<Point, 4>;

struct OCRPredictResult {
    Box box;
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

private:
    class PaddleOCRImpl;
    std::shared_ptr<PaddleOCRImpl> pImpl;
};

} // namespace PaddleOCR