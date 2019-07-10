find_package(TensorFlow REQUIRED)

add_library(cp_fused_nchw SHARED cp_fused_nchw.cc cp_fused_nchw.cu)
target_link_libraries(cudnn_full_conv2d PRIVATE cudart TensorFlow::TensorFlow)
