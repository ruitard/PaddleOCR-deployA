# PaddleOCR-deployA
本项目提供开箱即用的离线图片 OCR 文字识别程序。

本项目基于 [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR/) 进行开发，提高了 C++ 工程化质量，降低了开发部署难度。

- 使用 C++20 开发
- 构建工具链完全 CMake 化
- 支持 Linux 和 Windows 端
- 可自定义推理库和模型库

## 快速上手部署
本项目使用 [vcpkg](https://github.com/microsoft/vcpkg) 管理第三方开源库。在开始前请先配置好 `vcpkg`，并安装以下依赖库：
```shell
vcpkg install opencv openblas
```
### 下载 Paddle 推理库
> https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html

### 下载 Paddle 模型文件
> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/doc/doc_ch/models_list.md
> https://github.com/PaddlePaddle/PaddleOCR/blob/release/2.6/ppocr/utils/ppocr_keys_v1.txt

### 编译安装
```shell
cmake -G "Ninja Multi-Config" \
      -B build \
      -DCMAKE_TOOLCHAIN_FILE=/path/to/vcpkg/scripts/buildsystems/vcpkg.cmake \
      -DCMAKE_INSTALL_PREFIX=installed \
      -DPADDLE_LIB_DIRECTORY=/path/to/paddle_inference \
      -DPADDLE_MODEL_DIRECTORY=/path/to/paddle_model
cmake --build build --config Release --target install
cmake --build build --config Release --target package
```

