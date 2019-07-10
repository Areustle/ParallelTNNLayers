#
# Test
#
add_executable(test_main test/test_main.cpp.cu)
target_link_libraries(test_main PRIVATE doctest)

# ##############################################################################
# Tensor
# ##############################################################################
add_library(Tensor SHARED Tensor.cpp.cu)
target_include_directories(
  Tensor
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_link_libraries(Tensor PUBLIC doctest)
target_link_libraries(test_main PRIVATE Tensor)

# ##############################################################################
# cudnnConv2d
# ##############################################################################
add_library(cudnnConv2d SHARED cudnnConv2d.cpp.cu)
target_include_directories(
  cudnnConv2d
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_link_libraries(cudnnConv2d PUBLIC cudnn Tensor)
target_link_libraries(test_main PRIVATE cudnnConv2d)

# ##############################################################################
# Utils
# ##############################################################################
add_library(Utils SHARED Utils.cpp.cu)
target_include_directories(
  Utils
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_link_libraries(Utils PUBLIC Tensor)
target_link_libraries(test_main PRIVATE Utils)

# ##############################################################################
# Manual
# ##############################################################################
add_library(manual SHARED manual.cpp.cu)
target_include_directories(
  manual
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_link_libraries(manual PUBLIC Tensor)
target_link_libraries(test_main PRIVATE manual)

include(CTest)

# target_compile_features(test_cudnn_full_conv2d PRIVATE cxx_std_14)

# add_executable(manual manual.cu) target_compile_features(manual PRIVATE
# cxx_std_14)
