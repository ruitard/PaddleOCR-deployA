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

#pragma once

#include "ocr_cls.h"
#include "ocr_det.h"
#include "ocr_rec.h"

namespace PaddleOCR {

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

} // namespace PaddleOCR
