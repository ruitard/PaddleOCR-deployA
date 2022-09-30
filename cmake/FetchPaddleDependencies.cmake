add_library(paddle_inference SHARED IMPORTED GLOBAL)
add_library(paddle2onnx SHARED IMPORTED GLOBAL)
add_library(onnx_runtime SHARED IMPORTED GLOBAL)

if(UNIX)
set_target_properties(paddle_inference PROPERTIES
    IMPORTED_LOCATION ${PADDLE_LIB_DIRECTORY}/paddle/lib/libpaddle_inference.so
    INTERFACE_LINK_LIBRARIES "paddle2onnx;onnx_runtime"
)
set_target_properties(paddle2onnx PROPERTIES
    IMPORTED_LOCATION ${PADDLE_LIB_DIRECTORY}/third_party/install/paddle2onnx/lib/libpaddle2onnx.so
    IMPORTED_SONAME "libpaddle2onnx.so.1.0.0rc2"
)
set_target_properties(onnx_runtime PROPERTIES
    IMPORTED_LOCATION ${PADDLE_LIB_DIRECTORY}/third_party/install/onnxruntime/lib/libonnxruntime.so
    IMPORTED_SONAME "libonnxruntime.so.1.11.1"
)
endif()

if(WIN32)
set_target_properties(paddle_inference PROPERTIES
    IMPORTED_LOCATION ${PADDLE_LIB_DIRECTORY}/paddle/lib/paddle_inference.dll
    IMPORTED_IMPLIB ${PADDLE_LIB_DIRECTORY}/paddle/lib/paddle_inference.lib
)
set_target_properties(paddle2onnx PROPERTIES
    IMPORTED_LOCATION ${PADDLE_LIB_DIRECTORY}/third_party/install/paddle2onnx/lib/paddle2onnx.dll
)
set_target_properties(onnx_runtime PROPERTIES
    IMPORTED_LOCATION ${PADDLE_LIB_DIRECTORY}/third_party/install/onnxruntime/lib/onnxruntime.dll
)
endif()

target_include_directories(paddle_inference INTERFACE ${PADDLE_LIB_DIRECTORY}/paddle/include)
add_library(paddle::inference ALIAS paddle_inference)

install(IMPORTED_RUNTIME_ARTIFACTS paddle_inference paddle2onnx onnx_runtime)

install(DIRECTORY ${PADDLE_MODEL_DIRECTORY}/ DESTINATION bin)
