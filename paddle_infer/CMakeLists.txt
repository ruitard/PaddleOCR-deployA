find_package(OpenCV CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

add_executable(ppocra ${PPOCR_SOURCES})
add_subdirectory(src)

if(MSVC)
target_compile_definitions(ppocra PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(ppocra PRIVATE /utf-8)
endif()

target_link_libraries(ppocra
PRIVATE
    paddle::inference
    opencv_imgcodecs
    gflags::gflags
    spdlog::spdlog
)

set_target_properties(ppocra PROPERTIES
    INSTALL_RPATH "$ORIGIN:$ORIGIN/../lib"
)
install(TARGETS ppocra)
