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

#include <vector>
#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>

#include "paddle_infer.hpp"

namespace PaddleOCR {

namespace Utility {

std::vector<std::string> ReadDict(const fs::path &path);

template <class ForwardIterator>
inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
    return std::distance(first, std::max_element(first, last));
}

cv::Mat GetRotateCropImage(const cv::Mat &srcimage, const Box &box);

std::vector<int> argsort(const std::vector<float> &array);

void sorted_boxes(std::vector<OCRPredictResult> &ocr_result);

} // namespace Utility

} // namespace PaddleOCR