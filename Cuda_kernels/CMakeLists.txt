
# add_library(cudnn_full_conv2d SHARED cudnn_full_conv2d.cc.cu)
# target_link_libraries(cudnn_full_conv2d PUBLIC cudnn)
# target_compile_features(cudnn_full_conv2d PRIVATE cxx_std_14)


add_executable(test_cudnn_full_conv2d test_cudnn_full_conv2d.cu)
target_link_libraries(test_cudnn_full_conv2d PRIVATE cudnn_full_conv2d)
target_compile_features(test_cudnn_full_conv2d PRIVATE cxx_std_14)


add_executable(manual manual.cu)
target_compile_features(manual PRIVATE cxx_std_14)
