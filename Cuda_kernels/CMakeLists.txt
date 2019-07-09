#
# Test
#
add_executable(test_main test/test_main.cpp.cu)
target_link_libraries(test_main PRIVATE doctest)

# ##############################################################################
# Tensor
# ##############################################################################
add_library(Tensor SHARED src/Tensor.cpp.cu)
target_include_directories(
  Tensor
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_link_libraries(Tensor PUBLIC doctest)
target_link_libraries(test_main PRIVATE Tensor)

# ##############################################################################
# CUDNN_FULL_CONV2d
# ##############################################################################
add_library(cudnn_full_conv2d SHARED src/cudnn_full_conv2d.cpp.cu)
target_include_directories(
  cudnn_full_conv2d
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_link_libraries(cudnn_full_conv2d PUBLIC cudnn Tensor)
target_link_libraries(test_main PRIVATE cudnn_full_conv2d)

# ##############################################################################
# Conv_Utils
# ##############################################################################
add_library(conv_utils SHARED src/conv_utils.cpp.cu)
target_include_directories(
  conv_utils
  PUBLIC $<INSTALL_INTERFACE:include>
         $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_link_libraries(conv_utils PUBLIC Tensor)
target_link_libraries(test_main PRIVATE conv_utils)

# add_library(manual SHARED src/manual.cpp.cu) target_include_directories(manual
# PUBLIC $<INSTALL_INTERFACE:include>
# $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
# target_link_libraries(manual PUBLIC Tensor)

include(CTest)

# target_compile_features(test_cudnn_full_conv2d PRIVATE cxx_std_14)

# add_executable(manual manual.cu) target_compile_features(manual PRIVATE
# cxx_std_14)
