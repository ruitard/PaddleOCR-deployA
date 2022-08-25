cmake_minimum_required(VERSION 3.24)
project(ppocra VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded")
set(CMAKE_INSTALL_PREFIX ${PROJECT_BINARY_DIR}/installed)

include(CTest)
enable_testing()

include(cmake/FetchPaddleDependencies.cmake)

add_subdirectory(paddle_infer)
#add_subdirectory(src)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
