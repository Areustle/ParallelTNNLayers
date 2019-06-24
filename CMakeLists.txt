cmake_minimum_required(VERSION 3.12)

project(ParallelTNNLayers VERSION 0.2 LANGUAGES CXX CUDA DESCRIPTION "")

# Always use '-fPIC'/'-fPIE' option.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_subdirectory(external)
add_subdirectory(Cuda_kernels)
# add_subdirectory(Kernels)
