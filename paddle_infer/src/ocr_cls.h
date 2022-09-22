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

#include "paddle_api.h"
#include "paddle_inference_api.h"

#include "preprocess_op.h"
#include "utility.h"

namespace PaddleOCR {

class Classifier {
public:
    Classifier(const std::string &model_dir, unsigned int cpu_math_library_num_threads) {
        this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
        LoadModel(model_dir);
    }
  double cls_thresh = 0.9;

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir);

  void Run(std::vector<cv::Mat> img_list, std::vector<int> &cls_labels, std::vector<float> &cls_scores);

  private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  unsigned int cpu_math_library_num_threads_ = 4;

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
