# find_package(TensorFlow REQUIRED)

# add_library(cp_fused_nchw SHARED cp_fused_nchw.cc cp_fused_nchw.cu)
# target_link_libraries(cudnn_full_conv2d PRIVATE cudart TensorFlow::TensorFlow)
# target_compile_features(cp_fused_nchw PRIVATE cxx_std_11)

add_executable(cudnn_conv cudnn_full_conv2d.cu)
target_link_libraries(cudnn_conv PUBLIC cudnn)
add_executable(tvm_cp4_conv2d tvm_cp4_conv2d.cu)
target_link_libraries(tvm_cp4_conv2d PUBLIC cudnn)
