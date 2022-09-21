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

#include "ocr_cls.h"
#include "utility.h"

namespace PaddleOCR {

class CRNNRecognizer {
public:
    CRNNRecognizer(const std::string &model_dir, unsigned int cpu_math_library_num_threads,
                   const std::string &label_path, int rec_batch_num, int rec_img_h, int rec_img_w) {
        this->cpu_math_library_num_threads_ = cpu_math_library_num_threads;
        this->rec_batch_num_ = rec_batch_num;
        this->rec_img_h_ = rec_img_h;
        this->rec_img_w_ = rec_img_w;
        std::vector<int> rec_image_shape = {3, rec_img_h, rec_img_w};
        this->rec_image_shape_ = rec_image_shape;

        this->label_list_ = Utility::ReadDict(label_path);
        this->label_list_.insert(this->label_list_.begin(),
                                 "#"); // blank char for ctc
        this->label_list_.push_back(" ");

        LoadModel(model_dir);
    }

  // Load Paddle inference model
  void LoadModel(const std::string &model_dir);

  void Run(std::vector<cv::Mat> img_list, std::vector<std::string> &rec_texts, std::vector<float> &rec_text_scores);

  private:
  std::shared_ptr<paddle_infer::Predictor> predictor_;

  unsigned int cpu_math_library_num_threads_ = 4;

  std::vector<std::string> label_list_;

  std::vector<float> mean_ = {0.5f, 0.5f, 0.5f};
  std::vector<float> scale_ = {1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
  bool is_scale_ = true;
  bool use_tensorrt = false;
  int rec_batch_num_ = 6;
  int rec_img_h_ = 32;
  int rec_img_w_ = 320;
  std::vector<int> rec_image_shape_ = {3, rec_img_h_, rec_img_w_};
  // pre-process
  CrnnResizeImg resize_op_;
  Normalize normalize_op_;
  PermuteBatch permute_op_;

}; // class CrnnRecognizer

} // namespace PaddleOCR
