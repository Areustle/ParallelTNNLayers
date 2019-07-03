add_library(Tensor SHARED 
  src/Tensor.cpp.cu)

target_include_directories( Tensor
  PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cuda_kernels>)



add_executable(test_Tensor test/test_Tensor.cpp.cu)
target_link_libraries(test_Tensor PRIVATE Tensor doctest)


# add_library(cudnn_full_conv2d SHARED cudnn_full_conv2d.cpp.cu)
# target_link_libraries(cudnn_full_conv2d PUBLIC cudnn)
# target_compile_features(cudnn_full_conv2d PRIVATE cxx_std_14)

# target_compile_features(test_cudnn_full_conv2d PRIVATE cxx_std_14)

# add_executable(manual manual.cu) target_compile_features(manual PRIVATE
# cxx_std_14)
