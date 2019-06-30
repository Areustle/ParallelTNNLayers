add_library(CudaAllocator SHARED CudaAllocator.cpp.cu CudaAllocator.h)
target_compile_features(CudaAllocator PRIVATE cxx_std_14)

# add_library(cudnn_full_conv2d SHARED cudnn_full_conv2d.cpp.cu)
# target_link_libraries(cudnn_full_conv2d PUBLIC cudnn)
# target_compile_features(cudnn_full_conv2d PRIVATE cxx_std_14)

add_library(Tensor SHARED Tensor.cpp)

add_executable(test_convolution test_convolution.cpp)
target_link_libraries(test_convolution
                      PRIVATE CudaAllocator Tensor doctest)
# target_compile_features(test_cudnn_full_conv2d PRIVATE cxx_std_14)

# add_executable(manual manual.cu) target_compile_features(manual PRIVATE
# cxx_std_14)
