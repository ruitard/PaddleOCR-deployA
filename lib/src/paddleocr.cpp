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

void Classifier::Run(const std::vector<cv::Mat> &img_list, std::vector<int> &cls_labels,
                     std::vector<float> &cls_scores) {
    int img_num = img_list.size();
    std::vector<int> cls_image_shape = {3, 48, 192};
    for (int beg_img_no = 0; beg_img_no < img_num; beg_img_no += this->cls_batch_num) {
        int end_img_no = std::min(img_num, beg_img_no + this->cls_batch_num);
        int batch_num = end_img_no - beg_img_no;
        // preprocess
        std::vector<cv::Mat> norm_img_batch;
        for (int ino = beg_img_no; ino < end_img_no; ino++) {
            cv::Mat srcimg;
            img_list[ino].copyTo(srcimg);
            cv::Mat resize_img;
            this->resize_op.Run(srcimg, resize_img, this->use_tensorrt, cls_image_shape);

            this->normalize_op.Run(&resize_img, this->mean, this->scale, this->is_scale);
            if (resize_img.cols < cls_image_shape[2]) {
                cv::copyMakeBorder(resize_img, resize_img, 0, 0, 0,
                                   cls_image_shape[2] - resize_img.cols, cv::BORDER_CONSTANT,
                                   cv::Scalar(0, 0, 0));
            }
            norm_img_batch.push_back(resize_img);
        }
        std::vector<float> input(
            batch_num * cls_image_shape[0] * cls_image_shape[1] * cls_image_shape[2], 0.0f);
        this->permute_op.Run(norm_img_batch, input.data());

        // inference.
        auto input_names = this->predictor->GetInputNames();
        auto input_t = this->predictor->GetInputHandle(input_names[0]);
        input_t->Reshape({batch_num, cls_image_shape[0], cls_image_shape[1], cls_image_shape[2]});
        input_t->CopyFromCpu(input.data());
        this->predictor->Run();

        std::vector<float> predict_batch;
        auto output_names = this->predictor->GetOutputNames();
        auto output_t = this->predictor->GetOutputHandle(output_names[0]);
        auto predict_shape = output_t->shape();

        int out_num =
            std::accumulate(predict_shape.begin(), predict_shape.end(), 1, std::multiplies<int>());
        predict_batch.resize(out_num);

        output_t->CopyToCpu(predict_batch.data());

        // postprocess
        for (int batch_idx = 0; batch_idx < predict_shape[0]; batch_idx++) {
            int label = int(Utility::argmax(&predict_batch[batch_idx * predict_shape[1]],
                                            &predict_batch[(batch_idx + 1) * predict_shape[1]]));
            float score =
                float(*std::max_element(&predict_batch[batch_idx * predict_shape[1]],
                                        &predict_batch[(batch_idx + 1) * predict_shape[1]]));
            cls_labels[beg_img_no + batch_idx] = label;
            cls_scores[beg_img_no + batch_idx] = score;
        }
    }
}

void DBDetector::Run(const cv::Mat &img, std::vector<Box> &boxes) {
    float ratio_h{};
    float ratio_w{};

    cv::Mat srcimg;
    cv::Mat resize_img;
    img.copyTo(srcimg);

    this->resize_op.Run(img, resize_img, this->limit_type, this->limit_side_len, ratio_h, ratio_w,
                        this->use_tensorrt);

    this->normalize_op.Run(&resize_img, this->mean, this->scale, this->is_scale);

    std::vector<float> input(1 * 3 * resize_img.rows * resize_img.cols, 0.0f);
    this->permute_op.Run(&resize_img, input.data());

    // Inference.
    auto input_names = this->predictor->GetInputNames();
    auto input_t = this->predictor->GetInputHandle(input_names.front());
    input_t->Reshape({1, 3, resize_img.rows, resize_img.cols});
    input_t->CopyFromCpu(input.data());

    this->predictor->Run();

    std::vector<float> out_data;
    auto output_names = this->predictor->GetOutputNames();
    auto output_t = this->predictor->GetOutputHandle(output_names[0]);
    std::vector<int> output_shape = output_t->shape();
    int out_num =
        std::accumulate(output_shape.begin(), output_shape.end(), 1, std::multiplies<int>());

    out_data.resize(out_num);
    output_t->CopyToCpu(out_data.data());

    int n2 = output_shape[2];
    int n3 = output_shape[3];
    const int n = n2 * n3;

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

    boxes = post_processor.BoxesFromBitmap(pred_map, bit_map, this->det_db_box_thresh,
                                           this->det_db_unclip_ratio, this->det_db_score_mode);

    boxes = post_processor.FilterTagDetRes(boxes, ratio_h, ratio_w, srcimg);
}

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
            this->resize_op.Run(srcimg, resize_img, max_wh_ratio, this->use_tensorrt,
                                this->rec_image_shape);
            this->normalize_op.Run(&resize_img, this->mean, this->scale, this->is_scale);
            norm_img_batch.push_back(resize_img);
            batch_width = std::max(resize_img.cols, batch_width);
        }

        std::vector<float> input(batch_num * 3 * imgH * batch_width, 0.0f);
        this->permute_op.Run(norm_img_batch, input.data());

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

        int out_num =
            std::accumulate(predict_shape.begin(), predict_shape.end(), 1, std::multiplies<int>());
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
                argmax_idx = int(Utility::argmax(
                    &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                    &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));
                // get score
                max_value = float(*std::max_element(
                    &predict_batch[(m * predict_shape[1] + n) * predict_shape[2]],
                    &predict_batch[(m * predict_shape[1] + n + 1) * predict_shape[2]]));

                if (argmax_idx > 0 && (!(n > 0 && argmax_idx == last_index))) {
                    score += max_value;
                    count += 1;
                    str_res += label_list[argmax_idx];
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
