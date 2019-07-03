###############################################################################
# Libraries
###############################################################################
add_library(Tensor SHARED src/Tensor.cpp.cu)
add_library(cudnn_full_conv2d SHARED src/cudnn_full_conv2d.cpp.cu)
add_library(conv_utils SHARED src/conv_utils.cpp.cu)

###############################################################################
# Target Properties
###############################################################################

# target_include_directories
target_include_directories(Tensor PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_include_directories(cudnn_full_conv2d PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)
target_include_directories(conv_utils PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)

# # target_link_libraries
# target_link_libraries(Tensor PRIVATE)
target_link_libraries(cudnn_full_conv2d PUBLIC cudnn)
# target_link_libraries(conv_utils PRIVATE Tensor)

###############################################################################
# Tests
###############################################################################

add_executable(test_main
  test/test_main.cpp.cu
  test/test_cudnn_full_conv2d.cpp.cu
  test/test_Tensor.cpp.cu
  )

target_link_libraries(test_main PRIVATE 
  Tensor
  conv_utils
  cudnn_full_conv2d
  doctest
  )

include(CTest)

# target_compile_features(test_cudnn_full_conv2d PRIVATE cxx_std_14)

# add_executable(manual manual.cu) target_compile_features(manual PRIVATE
# cxx_std_14)
