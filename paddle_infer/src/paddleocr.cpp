// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <string_view>
#include <thread>

#include "ocr_cls.h"
#include "ocr_det.h"
#include "ocr_rec.h"

namespace PaddleOCR {

static const fs::path det_model_dir{"ch_PP-OCRv3_det_slim_infer"};
static const fs::path cls_model_dir{"ch_ppocr_mobile_v2.0_cls_slim_infer"};
static const fs::path rec_model_dir{"ch_PP-OCRv3_rec_slim_infer"};
static const fs::path rec_char_dict_path{"ppocr_keys_v1.txt"};

class PPOCR {
public:
    PPOCR();
    PPOCR(const PPOCR &) = delete;
    PPOCR(PPOCR &&) = delete;
    PPOCR &operator=(const PPOCR &) = delete;
    PPOCR &operator=(PPOCR &&) = delete;
    std::vector<OCRPredictResult> ocr(const cv::Mat &img, bool enable_cls = false);

private:
    void det(const cv::Mat &img, std::vector<OCRPredictResult> &ocr_results);
    void rec(const std::vector<cv::Mat> &img_list, std::vector<OCRPredictResult> &ocr_results);
    void cls(const std::vector<cv::Mat> &img_list, std::vector<OCRPredictResult> &ocr_results);

    std::unique_ptr<DBDetector> detector;
    std::unique_ptr<Classifier> classifier;
    std::unique_ptr<CRNNRecognizer> recognizer;
};

class PaddleOCR::PaddleOCRImpl {
public:
    std::unique_ptr<PPOCR> ppocr;
};

PaddleOCR::PaddleOCR() {
    pImpl = std::make_shared<PaddleOCRImpl>();
    pImpl->ppocr = std::make_unique<PPOCR>();
}

std::vector<OCRPredictResult> PaddleOCR::ocr(const fs::path &image_path) {
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    return pImpl->ppocr->ocr(img);
}

PPOCR::PPOCR() {
    auto cpu_threads = std::thread::hardware_concurrency();
    detector = std::make_unique<DBDetector>(det_model_dir, cpu_threads);
    classifier = std::make_unique<Classifier>(cls_model_dir, cpu_threads);
    recognizer = std::make_unique<CRNNRecognizer>(rec_model_dir, cpu_threads, rec_char_dict_path);
};

std::vector<OCRPredictResult> PPOCR::ocr(const cv::Mat &img, bool enable_cls) {
    std::vector<OCRPredictResult> ocr_results;

    this->det(img, ocr_results);
    // crop image
    std::vector<cv::Mat> img_list;
    for (const auto &result : ocr_results) {
        img_list.push_back(Utility::GetRotateCropImage(img, result.box));
    }

    // cls
    if (enable_cls && this->classifier != nullptr) {
        this->cls(img_list, ocr_results);
        for (int i = 0; i < img_list.size(); i++) {
            if (ocr_results[i].cls_label % 2 == 1 &&
                ocr_results[i].cls_score > this->classifier->cls_thresh) {
                cv::rotate(img_list[i], img_list[i], 1);
            }
        }
    }

    this->rec(img_list, ocr_results);

    return ocr_results;
}

void PPOCR::det(const cv::Mat &img, std::vector<OCRPredictResult> &ocr_results) {
    std::vector<std::vector<std::vector<int>>> boxes;

    this->detector->Run(img, boxes);

    for (int i = 0; i < boxes.size(); i++) {
        OCRPredictResult res;
        res.box = boxes[i];
        ocr_results.push_back(res);
    }
    // sort boex from top to bottom, from left to right
    Utility::sorted_boxes(ocr_results);
}

void PPOCR::rec(const std::vector<cv::Mat> &img_list, std::vector<OCRPredictResult> &ocr_results) {
    std::vector<std::string> rec_texts(img_list.size(), "");
    std::vector<float> rec_text_scores(img_list.size(), 0);
    this->recognizer->Run(img_list, rec_texts, rec_text_scores);
    // output rec results
    for (int i = 0; i < rec_texts.size(); i++) {
        ocr_results[i].text = rec_texts[i];
        ocr_results[i].score = rec_text_scores[i];
    }
}

void PPOCR::cls(const std::vector<cv::Mat> &img_list, std::vector<OCRPredictResult> &ocr_results) {
    std::vector<int> cls_labels(img_list.size(), 0);
    std::vector<float> cls_scores(img_list.size(), 0);
    this->classifier->Run(img_list, cls_labels, cls_scores);
    // output cls results
    for (int i = 0; i < cls_labels.size(); i++) {
        ocr_results[i].cls_label = cls_labels[i];
        ocr_results[i].cls_score = cls_scores[i];
    }
}

} // namespace PaddleOCR
