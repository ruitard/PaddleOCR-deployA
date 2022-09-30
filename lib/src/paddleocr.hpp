#pragma once

#include <thread>
#include <paddle_inference_api.h>

#include "preprocess_op.h"
#include "postprocess_op.h"

#include "paddle_infer.hpp"

namespace PaddleOCR {

inline auto create_predictor(const fs::path &model_path)
    -> std::shared_ptr<paddle_infer::Predictor> {
    paddle_infer::Config config;
    config.SetModel((model_path / "inference.pdmodel").string(),
                    (model_path / "inference.pdiparams").string());
    config.DisableGpu();
    config.SetCpuMathLibraryNumThreads(std::thread::hardware_concurrency());

    auto pass_builder = config.pass_builder();
    pass_builder->DeletePass("conv_transpose_eltwiseadd_bn_fuse_pass");
    pass_builder->DeletePass("matmul_transpose_reshape_fuse_pass");

    // false for zero copy tensor
    config.SwitchUseFeedFetchOps(false);
    // true for multiple input
    config.SwitchSpecifyInputNames(true);

    config.SwitchIrOptim(true);

    config.EnableMemoryOptim();
    config.DisableGlogInfo();

    return paddle_infer::CreatePredictor(config);
}

class Classifier {
public:
    explicit Classifier(const fs::path &model_path) : predictor{create_predictor(model_path)} {}
    double cls_thresh = 0.9;
    void Run(const std::vector<cv::Mat> &img_list, std::vector<int> &cls_labels,
             std::vector<float> &cls_scores);

private:
    std::shared_ptr<paddle_infer::Predictor> predictor;

    std::array<float, 3> mean{0.5f, 0.5f, 0.5f};
    std::array<float, 3> scale{1 / 0.5f, 1 / 0.5f, 1 / 0.5f};
    bool is_scale = true;
    bool use_tensorrt = false;
    int cls_batch_num = 1;
    // pre-process
    ClsResizeImg resize_op;
    Normalize normalize_op;
    PermuteBatch permute_op;
};

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

    std::array<float, 3> mean{0.485f, 0.456f, 0.406f};
    std::array<float, 3> scale{1 / 0.229f, 1 / 0.224f, 1 / 0.225f};
    bool is_scale = true;

    // pre-process
    ResizeImgType0 resize_op;
    Normalize normalize_op;
    Permute permute_op;

    // post-process
    DBPostProcessor post_processor;
};

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
};

} // namespace PaddleOCR