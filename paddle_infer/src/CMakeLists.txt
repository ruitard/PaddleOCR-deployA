target_sources(ppocra
PRIVATE
    clipper.cpp
    ocr_cls.cpp ocr_det.cpp ocr_rec.cpp
    paddleocr.cpp
    postprocess_op.cpp preprocess_op.cpp utility.cpp
)
target_include_directories(ppocra PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})