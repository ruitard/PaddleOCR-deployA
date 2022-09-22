target_sources(paddleocr
PRIVATE
    clipper.cpp
    ocr_cls.cpp ocr_det.cpp ocr_rec.cpp
    paddleocr.cpp
    postprocess_op.cpp preprocess_op.cpp utility.cpp
)
target_include_directories(paddleocr PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})