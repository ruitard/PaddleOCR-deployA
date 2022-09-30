#pragma once

#include <thread>
#include <paddle_inference_api.h>

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

} // namespace PaddleOCR