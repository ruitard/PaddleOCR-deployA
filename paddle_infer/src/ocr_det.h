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

#include "postprocess_op.h"
#include "preprocess_op.h"

namespace PaddleOCR {

class DBDetector {
public:
    explicit DBDetector(const fs::path &model_path) : predictor{create_predictor(model_path)} {}

    void Run(const cv::Mat &img, std::vector<Box> &boxes);

private:
    std::shared_ptr<paddle_infer::Predictor> predictor;

    std::string limit_type = "max";
    int limit_side_len = 960;

    double det_db_thresh = 0.3;
    double det_db_box_thresh = 0.6;
    double det_db_unclip_ratio = 1.5;
    std::string det_db_score_mode = "slow";
    bool use_dilation = false;

    bool use_tensorrt = false;

    std::vector<float> mean_ = {0.485f, 0.456f, 0.406f};
    std::vector<float> scale_ = {1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    bool is_scale_ = true;

    // pre-process
    ResizeImgType0 resize_op_;
    Normalize normalize_op_;
    Permute permute_op_;

    // post-process
    DBPostProcessor post_processor_;
};

} // namespace PaddleOCR