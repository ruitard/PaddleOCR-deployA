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
#include "utility.hpp"

namespace PaddleOCR {

class CRNNRecognizer {
public:
    CRNNRecognizer(const fs::path &model_path, const fs::path &label_path) :
        predictor{create_predictor(model_path)} {

        this->label_list = Utility::ReadDict(label_path);
        this->label_list.insert(this->label_list.begin(), "#"); // blank char for ctc
        this->label_list.push_back(" ");
    }
    void Run(const std::vector<cv::Mat> &img_list, std::vector<std::string> &rec_texts,
             std::vector<float> &rec_text_scores);

private:
    std::shared_ptr<paddle_infer::Predictor> predictor;

    std::vector<std::string> label_list;

    std::array<float, 3> mean{0.5f, 0.5f, 0.5f};
    std::array<float, 3> scale{1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    bool is_scale = true;
    bool use_tensorrt = false;
    int rec_batch_num = 6;
    int rec_img_h = 48;
    int rec_img_w = 320;
    std::array<int, 3> rec_image_shape{3, 48, 320};
    // pre-process
    CrnnResizeImg resize_op;
    Normalize normalize_op;
    PermuteBatch permute_op;

}; // class CrnnRecognizer

} // namespace PaddleOCR
