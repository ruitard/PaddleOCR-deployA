cmake_minimum_required(VERSION 3.24)
project(ppocra VERSION 1.0.0 HOMEPAGE_URL "https://github.com/ruitard/PaddleOCR-deployA")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
#set(CMAKE_COMPILE_WARNING_AS_ERROR ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_LIBDIR lib)

if(BUILD_WITH_ASAN)
    add_compile_options(-fsanitize=address)
    add_link_options(-fsanitize=address)
endif()

if(BUILD_WITH_UBSAN)
    add_compile_options(-fsanitize=undefined)
    add_link_options(-fsanitize=undefined)
endif()

include(CTest)
enable_testing()

find_package(Threads REQUIRED)

include(cmake/FetchPaddleDependencies.cmake)

add_subdirectory(paddle_infer)
add_subdirectory(cli)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
