cmake_minimum_required(VERSION 3.12)

project(
  ParallelTNNLayers
  VERSION 0.2
  LANGUAGES CUDA CXX
  DESCRIPTION "A collection of Decomposed Tensor Operations for use in TNNs.")

# Always use '-fPIC'/'-fPIE' option.
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(external)
add_subdirectory(Cuda_kernels)
add_subdirectory(Kernels)
add_subdirectory(utils)
