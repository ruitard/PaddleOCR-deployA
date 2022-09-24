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

#include "preprocess_op.h"
#include "utility.hpp"

namespace PaddleOCR {

class Classifier {
public:
    explicit Classifier(const fs::path &model_path) : predictor{create_predictor(model_path)} {}
    double cls_thresh = 0.9;
    void Run(const std::vector<cv::Mat> &img_list, std::vector<int> &cls_labels,
             std::vector<float> &cls_scores);

private:
    std::shared_ptr<paddle_infer::Predictor> predictor;

    std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
    std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    bool is_scale_ = true;
    bool use_tensorrt = false;
    int cls_batch_num = 1;
    // pre-process
    ClsResizeImg resize_op_;
    Normalize normalize_op_;
    PermuteBatch permute_op_;

}; // class Classifier

} // namespace PaddleOCR
