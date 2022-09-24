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

#include "utility.hpp"
#include <iostream>
#include <fstream>
#include <vector>

namespace PaddleOCR {

std::vector<std::string> Utility::ReadDict(const fs::path &path) {
    std::vector<std::string> label_vec;
    if (std::ifstream ifs{path}; ifs.is_open()) {
        std::string line;
        while (std::getline(ifs, line)) {
            label_vec.push_back(line);
        }
    } else {
        throw std::runtime_error("no such label file: " + path.string());
    }
    return label_vec;
}

cv::Mat Utility::GetRotateCropImage(const cv::Mat &srcimage, const Box &box) {
    cv::Mat image;
    srcimage.copyTo(image);
    Box points = box;

    int x_collect[4] = {box[0][0], box[1][0], box[2][0], box[3][0]};
    int y_collect[4] = {box[0][1], box[1][1], box[2][1], box[3][1]};
    int left = int(*std::min_element(x_collect, x_collect + 4));
    int right = int(*std::max_element(x_collect, x_collect + 4));
    int top = int(*std::min_element(y_collect, y_collect + 4));
    int bottom = int(*std::max_element(y_collect, y_collect + 4));

    cv::Mat img_crop;
    image(cv::Rect(left, top, right - left, bottom - top)).copyTo(img_crop);

    for (int i = 0; i < points.size(); i++) {
        points[i][0] -= left;
        points[i][1] -= top;
    }

    int img_crop_width =
        int(sqrt(pow(points[0][0] - points[1][0], 2) + pow(points[0][1] - points[1][1], 2)));
    int img_crop_height =
        int(sqrt(pow(points[0][0] - points[3][0], 2) + pow(points[0][1] - points[3][1], 2)));

    cv::Point2f pts_std[4];
    pts_std[0] = cv::Point2f(0., 0.);
    pts_std[1] = cv::Point2f(img_crop_width, 0.);
    pts_std[2] = cv::Point2f(img_crop_width, img_crop_height);
    pts_std[3] = cv::Point2f(0.f, img_crop_height);

    cv::Point2f pointsf[4];
    pointsf[0] = cv::Point2f(points[0][0], points[0][1]);
    pointsf[1] = cv::Point2f(points[1][0], points[1][1]);
    pointsf[2] = cv::Point2f(points[2][0], points[2][1]);
    pointsf[3] = cv::Point2f(points[3][0], points[3][1]);

    cv::Mat M = cv::getPerspectiveTransform(pointsf, pts_std);

    cv::Mat dst_img;
    cv::warpPerspective(img_crop, dst_img, M, cv::Size(img_crop_width, img_crop_height),
                        cv::BORDER_REPLICATE);

    if (float(dst_img.rows) >= float(dst_img.cols) * 1.5) {
        cv::Mat srcCopy = cv::Mat(dst_img.rows, dst_img.cols, dst_img.depth());
        cv::transpose(dst_img, srcCopy);
        cv::flip(srcCopy, srcCopy, 0);
        return srcCopy;
    } else {
        return dst_img;
    }
}

std::vector<int> Utility::argsort(const std::vector<float> &array) {
    const int array_len(array.size());
    std::vector<int> array_index(array_len, 0);
    for (int i = 0; i < array_len; ++i)
        array_index[i] = i;

    std::sort(array_index.begin(), array_index.end(),
              [&array](int pos1, int pos2) { return (array[pos1] < array[pos2]); });

    return array_index;
}

static bool comparison_box(const OCRPredictResult &result1, const OCRPredictResult &result2) {
    if (result1.box[0][1] < result2.box[0][1]) {
        return true;
    } else if (result1.box[0][1] == result2.box[0][1]) {
        return result1.box[0][0] < result2.box[0][0];
    } else {
        return false;
    }
}

void Utility::sorted_boxes(std::vector<OCRPredictResult> &ocr_result) {
    std::sort(ocr_result.begin(), ocr_result.end(), comparison_box);
    if (ocr_result.size() > 0) {
        for (int i = 0; i < ocr_result.size() - 1; i++) {
            for (int j = i; j > 0; j--) {
                if (abs(ocr_result[j + 1].box[0][1] - ocr_result[j].box[0][1]) < 10 &&
                    (ocr_result[j + 1].box[0][0] < ocr_result[j].box[0][0])) {
                    std::swap(ocr_result[i], ocr_result[i + 1]);
                }
            }
        }
    }
}

} // namespace PaddleOCR