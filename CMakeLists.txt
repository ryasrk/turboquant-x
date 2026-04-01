cmake_minimum_required(VERSION 3.20)
project(turboquant_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Optimization flags
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")

# OpenMP for thread parallelism
find_package(OpenMP)

# Find pybind11
find_package(pybind11 REQUIRED)

# Build the extension module
pybind11_add_module(_turboquant_cpp
    src/turboquant_cpp/rotation.cpp
    src/turboquant_cpp/codebook.cpp
    src/turboquant_cpp/polar_quant.cpp
    src/turboquant_cpp/bindings.cpp
)

target_include_directories(_turboquant_cpp PRIVATE
    ${CMAKE_SOURCE_DIR}/src/turboquant_cpp
)

# Link OpenMP if available
if(OpenMP_CXX_FOUND)
    target_link_libraries(_turboquant_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# Install to the src/turboquant_cpp directory so Python can import it
install(TARGETS _turboquant_cpp DESTINATION src/turboquant_cpp)
