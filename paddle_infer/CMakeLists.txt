find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

file(GLOB PPOCR_SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${PPOCR_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(${PROJECT_NAME}
PRIVATE
    paddle::inference
    opencv_imgcodecs
    gflags::gflags
    spdlog::spdlog
)

set_target_properties(${PROJECT_NAME} PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)
install(TARGETS ${PROJECT_NAME})
