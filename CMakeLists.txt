cmake_minimum_required(VERSION 3.24)
project(ppocra VERSION 0.0.1)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_INSTALL_LIBDIR lib)
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/installed)

include(CTest)
enable_testing()

find_package(Threads REQUIRED)

include(cmake/FetchPaddleDependencies.cmake)

add_subdirectory(paddle_infer)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
