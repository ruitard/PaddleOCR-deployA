find_package(OpenCV CONFIG REQUIRED)

add_library(paddleocr)
target_include_directories(paddleocr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(src)

if(MSVC)
target_compile_definitions(paddleocr PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(paddleocr PRIVATE /utf-8)
endif()

target_link_libraries(paddleocr
PRIVATE
    paddle::inference
    opencv_imgcodecs
)
