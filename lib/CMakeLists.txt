find_package(OpenCV CONFIG REQUIRED)

add_library(paddleocr)
target_include_directories(paddleocr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
add_subdirectory(src)

if(MSVC)
target_compile_definitions(paddleocr PRIVATE WIN32_LEAN_AND_MEAN NOMINMAX)
target_compile_options(paddleocr PRIVATE /utf-8 /openmp)
target_link_options(paddleocr PRIVATE /RELEASE)
endif()

target_link_libraries(paddleocr
PRIVATE
    paddle::inference
    opencv_imgcodecs
)

if(MSVC)
find_package(OpenBLAS CONFIG REQUIRED)

if(BUILD_SHARED_LIBS)
install(FILES $<TARGET_PDB_FILE:paddleocr> DESTINATION bin OPTIONAL)
endif()

install(IMPORTED_RUNTIME_ARTIFACTS opencv_imgcodecs OpenBLAS::OpenBLAS)
install(CODE [[
file(GET_RUNTIME_DEPENDENCIES
    RESOLVED_DEPENDENCIES_VAR RESOLVED_DEPS
    UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPS
    LIBRARIES $<TARGET_FILE:opencv_imgcodecs>
    DIRECTORIES $<TARGET_FILE_DIR:opencv_imgcodecs>
    PRE_INCLUDE_REGEXES $<TARGET_FILE_DIR:opencv_imgcodecs>
    PRE_EXCLUDE_REGEXES "system32"
    POST_INCLUDE_REGEXES $<TARGET_FILE_DIR:opencv_imgcodecs>
    POST_EXCLUDE_REGEXES "system32"
)
foreach(DEP_LIB ${RESOLVED_DEPS})
    file(INSTALL ${DEP_LIB} DESTINATION ${CMAKE_INSTALL_PREFIX}/bin)
endforeach()
]])
endif()