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

#include "ocr_det.h"

namespace PaddleOCR {

void DBDetector::LoadModel(const std::string &model_dir) {
    paddle_infer::Config config;
    config.SetModel(model_dir + "/inference.pdmodel", model_dir + "/inference.pdiparams");

    config.DisableGpu();
    config.SetCpuMathLibraryNumThreads(this->cpu_math_library_num_threads_);

    // use zero_copy_run as default
    config.SwitchUseFeedFetchOps(false);
    // true for multiple input
    config.SwitchSpecifyInputNames(true);

    config.SwitchIrOptim(true);

    config.EnableMemoryOptim();
    config.DisableGlogInfo();

    this->predictor_ = paddle_infer::CreatePredictor(config);
}

void DBDetector::Run(const cv::Mat &img, std::vector<std::vector<std::vector<int>>> &boxes) {
    float ratio_h{};
    float ratio_w{};

    cv::Mat srcimg;
    cv::Mat resize_img;
    img.copyTo(srcimg);

    this->resize_op_.Run(img, resize_img, this->limit_type, this->limit_side_len, ratio_h, ratio_w,
                         this->use_tensorrt);

    this->normalize_op_.Run(&resize_img, this->mean_, this->scale_, this->is_scale_);

    std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
    this->permute_op_.Run(&resize_img, input.data());

    // Inference.
    auto input_names = this->predictor_->GetInputNames();
    auto input_t = this->predictor_->GetInputHandle(input_names[0]);
    input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
    input_t->CopyFromCpu(input.data());

    this->predictor_->Run();

    std::vector<float> out_data;
    auto output_names = this->predictor_->GetOutputNames();
    auto output_t = this->predictor_->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num =
        std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());

    int n2 = output_shape[2];
    int n3 = output_shape[3];
    int n = n2 * n3;

    std::vector<float> pred(n, 0.0);
    std::vector<unsigned char> cbuf(n, ' ');

    for (int i = 0; i < n; i++) {
        pred[i] = float(out_data[i]);
        cbuf[i] = (unsigned char)((out_data[i]) * 255);
    }

    cv::Mat cbuf_map(n2, n3, CV_8UC1, (unsigned char *)cbuf.data());
    cv::Mat pred_map(n2, n3, CV_32F, (float *)pred.data());

    const double threshold = this->det_db_thresh * 255;
    const double maxvalue = 255;
    cv::Mat bit_map;
    cv::threshold(cbuf_map, bit_map, threshold, maxvalue, cv::THRESH_BINARY);
    if (this->use_dilation) {
        cv::Mat dila_ele = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
        cv::dilate(bit_map, bit_map, dila_ele);
    }

    boxes = post_processor_.BoxesFromBitmap(pred_map, bit_map, this->det_db_box_thresh,
                                            this->det_db_unclip_ratio, this->det_db_score_mode);

    boxes = post_processor_.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
}

} // namespace PaddleOCR
