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

#include "ocr_cls.h"
#include "ocr_det.h"
#include "ocr_rec.h"
#include "paddleocr.hpp"

namespace PaddleOCR {

class PaddleOCR::PaddleOCRImpl {
public:
    explicit PaddleOCRImpl(const PaddleConfig &config) {
        detector = std::make_unique<DBDetector>(config.det_model_dir);
        classifier = std::make_unique<Classifier>(config.cls_model_dir);
        recognizer =
            std::make_unique<CRNNRecognizer>(config.rec_model_dir, config.rec_char_dict_path);
    }
    std::vector<OCRPredictResult> ocr(const cv::Mat &img, bool enable_cls = false) const;

private:
    void det(const cv::Mat &img, std::vector<OCRPredictResult> &ocr_results) const;
    void rec(const std::vector<cv::Mat> &img_list,
             std::vector<OCRPredictResult> &ocr_results) const;
    void cls(const std::vector<cv::Mat> &img_list,
             std::vector<OCRPredictResult> &ocr_results) const;

    std::unique_ptr<DBDetector> detector;
    std::unique_ptr<Classifier> classifier;
    std::unique_ptr<CRNNRecognizer> recognizer;
};

PaddleOCR::PaddleOCR(const PaddleConfig &config) {
    pImpl = std::make_shared<PaddleOCRImpl>(config);
}

std::vector<OCRPredictResult> PaddleOCR::ocr(const fs::path &image_path) const {
    cv::Mat img = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    return pImpl->ocr(img);
}

std::vector<OCRPredictResult> PaddleOCR::PaddleOCRImpl::ocr(const cv::Mat &img,
                                                            bool enable_cls) const {
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

void PaddleOCR::PaddleOCRImpl::det(const cv::Mat &img,
                                   std::vector<OCRPredictResult> &ocr_results) const {
    std::vector<Box> boxes;

    this->detector->Run(img, boxes);

    for (int i = 0; i < boxes.size(); i++) {
        OCRPredictResult res;
        res.box = boxes[i];
        ocr_results.push_back(res);
    }
    // sort boex from top to bottom, from left to right
    Utility::sorted_boxes(ocr_results);
}

void PaddleOCR::PaddleOCRImpl::rec(const std::vector<cv::Mat> &img_list,
                                   std::vector<OCRPredictResult> &ocr_results) const {
    std::vector<std::string> rec_texts(img_list.size(), "");
    std::vector<float> rec_text_scores(img_list.size(), 0);
    this->recognizer->Run(img_list, rec_texts, rec_text_scores);
    // output rec results
    for (int i = 0; i < rec_texts.size(); i++) {
        ocr_results[i].text = rec_texts[i];
        ocr_results[i].score = rec_text_scores[i];
    }
}

void PaddleOCR::PaddleOCRImpl::cls(const std::vector<cv::Mat> &img_list,
                                   std::vector<OCRPredictResult> &ocr_results) const {
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
