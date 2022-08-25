# https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html

add_library(paddle2onnx SHARED IMPORTED GLOBAL)
set_target_properties(paddle2onnx PROPERTIES
    IMPORTED_LOCATION ${PADDLE_DIRECTORY}/third_party/install/paddle2onnx/lib/libpaddle2onnx.so
    IMPORTED_SONAME "libpaddle2onnx.so.1.0.0rc2"
)
install(IMPORTED_RUNTIME_ARTIFACTS paddle2onnx)

add_library(iomp SHARED IMPORTED GLOBAL)
set_target_properties(iomp PROPERTIES
    IMPORTED_LOCATION ${PADDLE_DIRECTORY}/third_party/install/mklml/lib/libiomp5.so
)
install(IMPORTED_RUNTIME_ARTIFACTS iomp)

add_library(mkl_ml SHARED IMPORTED GLOBAL)
set_target_properties(mkl_ml PROPERTIES
    IMPORTED_LOCATION ${PADDLE_DIRECTORY}/third_party/install/mklml/lib/libmklml_intel.so
    INTERFACE_LINK_LIBRARIES "iomp"
)
install(IMPORTED_RUNTIME_ARTIFACTS mkl_ml)

add_library(mkl_dnn SHARED IMPORTED GLOBAL)
set_target_properties(mkl_dnn PROPERTIES
    IMPORTED_LOCATION ${PADDLE_DIRECTORY}/third_party/install/mkldnn/lib/libmkldnn.so.0
    IMPORTED_SONAME "libdnnl.so.2"
)
install(IMPORTED_RUNTIME_ARTIFACTS mkl_dnn)

add_library(onnx_runtime SHARED IMPORTED GLOBAL)
set_target_properties(onnx_runtime PROPERTIES
    IMPORTED_LOCATION ${PADDLE_DIRECTORY}/third_party/install/onnxruntime/lib/libonnxruntime.so
    IMPORTED_SONAME "libonnxruntime.so.1.11.1"
)
install(IMPORTED_RUNTIME_ARTIFACTS onnx_runtime)

add_library(paddle_inference SHARED IMPORTED GLOBAL)
set_target_properties(paddle_inference PROPERTIES
    IMPORTED_LOCATION ${PADDLE_DIRECTORY}/paddle/lib/libpaddle_inference.so
    INTERFACE_LINK_LIBRARIES "paddle2onnx;mkl_dnn;onnx_runtime;iomp"
)
target_include_directories(paddle_inference INTERFACE ${PADDLE_DIRECTORY}/paddle/include)
install(IMPORTED_RUNTIME_ARTIFACTS paddle_inference)
add_library(paddle::inference ALIAS paddle_inference)