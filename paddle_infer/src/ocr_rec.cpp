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

#include "ocr_rec.h"

namespace PaddleOCR {

void CRNNRecognizer::Run(const std::vector<cv::Mat> &img_list, std::vector<std::string> &rec_texts,
                         std::vector<float> &rec_text_scores) {

    int img_num = img_list.size();
    std::vector<float> width_list;
    for (int i = 0; i < img_num; i++) {
        width_list.push_back(float(img_list[i].cols) / img_list[i].rows);
    }
    std::vector<int> indices = Utility::argsort(width_list);

    for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += this->rec_batch_num) {
        int end_img_no = std::min(img_num, beg_img_no + this->rec_batch_num);
        int batch_num = end_img_no - beg_img_no;
        int imgH = this->rec_image_shape[1];
        int imgW = this->rec_image_shape[2];
        float max_wh_ratio = imgW * 1.0 / imgH;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            int h = img_list[indices[ino]].rows;
            int w = img_list[indices[ino]].cols;
            float wh_ratio = w * 1.0 / h;
            max_wh_ratio = std::max(max_wh_ratio, wh_ratio);
        }

        int batch_width = imgW;
        std::vector<cv::Mat> norm_img_batch;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[indices[ino]].copyTo(srcimg);
            cv::Mat resize_img;
            this->resize_op_.Run(srcimg, resize_img, max_wh_ratio, this->use_tensorrt,
                                 this->rec_image_shape);
            this->normalize_op_.Run(&resize_img, this->mean_, this->scale_, this->is_scale_);
            norm_img_batch.push_back(resize_img);
            batch_width = std::max(resize_img.cols, batch_width);
        }

        std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
        this->permute_op_.Run(norm_img_batch, input.data());

        // Inference.
        auto input_names = this->predictor->GetInputNames();
        auto input_t = this->predictor->GetInputHandle(input_names[0]);
        input_t->Reshape({batch_num, 3, imgH, batch_width});
        input_t->CopyFromCpu(input.data());
        this->predictor->Run();

        std::vector<float> predict_batch;
        auto output_names = this->predictor->GetOutputNames();
        auto output_t = this->predictor->GetOutputHandle(output_names[0]);
        auto predict_shape = output_t->shape();

        int out_num = std::accumulate(predict_shape.begin(), predict_shape.end(), 1, std::multiplies<int>());
        predict_batch.resize(out_num);
        // predict_batch is the result of Last FC with softmax
        output_t->CopyToCpu(predict_batch.data());

        // ctc decode
        for (int m = 0; m < predict_shape[0]; m++) {
            std::string str_res;
            int argmax_idx;
            int last_index = 0;
            float score = 0.f;
            int count = 0;
            float max_value = 0.0f;

            for (int n = 0; n < predict_shape[1]; n++) {
                // get idx
                argmax_idx = int(Utility::argmax(&predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                                                 &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                // get score
                max_value = float(*std::max_element(&predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                                                    &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                    score += max_value;
                    count += 1;
                    str_res += label_list_[argmax_idx];
                }
                last_index = argmax_idx;
            }
            score /= count;
            if (std::isnan(score)) {
                continue;
            }
            rec_texts[indices[beg_img_no + m]] = str_res;
            rec_text_scores[indices[beg_img_no + m]] = score;
        }
    }
}

} // namespace PaddleOCR
