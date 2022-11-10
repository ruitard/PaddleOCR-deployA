# PaddleOCR-deployA
本项目提供开箱即用的离线图片 OCR 文字识别程序。

本项目基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/) 进行开发，提高了 C++ 工程化质量，降低了开发部署难度。

- 使用 C++20 开发
- 完全使用 CMake 进行构建
- 支持 Linux 和 Windows 端
- 可自定义推理库和模型库

## 快速上手部署
本项目使用 [vcpkg](https://github.com/microsoft/vcpkg) 管理第三方开源库。在开始前请先配置好 `vcpkg`，并安装以下依赖库：
```shell
vcpkg install opencv openblas
```
### 下载 Paddle 推理库
> https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html

本项目默认使用的是：
> https://paddle-inference-lib.bj.bcebos.com/2.3.2/cxx_c/Linux/CPU/gcc5.4_avx_openblas/paddle_inference.tgz

> https://paddle-inference-lib.bj.bcebos.com/2.3.2/cxx_c/Windows/CPU/x86-64_avx-openblas-vs2017/paddle_inference.zip

下载后解压，目录路径 CMake 配置时传给 `PADDLE_LIB_DIRECTORY`

GPU 版本后续考虑适配

### 下载 Paddle 模型文件
> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md

本项目默认使用：

中文检测模型：https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_det_slim_infer.tar

中文识别模型：https://paddleocr.bj.bcebos.com/PP-OCRv3/chinese/ch_PP-OCRv3_rec_slim_infer.tar

文本方向分类模型：https://paddleocr.bj.bcebos.com/dygraph_v2.0/ch/ch_ppocr_mobile_v2.0_cls_slim_infer.tar

以及中文词典文件：
> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt

将上述文件解包后置于同一路径下，如下所示，在 CMake 配置时传给 `PADDLE_MODEL_DIRECTORY`
```shell
paddle_model/
├── ch_ppocr_mobile_v2.0_cls_slim_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   ├── inference.pdmodel
│   └── paddle_infer.log
├── ch_PP-OCRv3_det_slim_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
├── ch_PP-OCRv3_rec_slim_infer
│   ├── inference.pdiparams
│   ├── inference.pdiparams.info
│   └── inference.pdmodel
└── ppocr_keys_v1.txt
```

### 编译安装
```shell
cmake -G "Ninja Multi-Config" \
      -B build \
      -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DCMAKE_INSTALL_PREFIX=installed \
      -DPADDLE_LIB_DIRECTORY=/path/to/paddle_inference \
      -DPADDLE_MODEL_DIRECTORY=/path/to/paddle_model \
      .
cmake --build build --config Release --target install
cmake --build build --config Release --target package
```
