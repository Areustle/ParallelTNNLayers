add_executable(manual manual.cu)
target_compile_features(manual PUBLIC cxx_std_11)

add_executable(cudnn_full_conv2d cudnn_full_conv2d.cu)
target_link_libraries(cudnn_full_conv2d PRIVATE cudnn)
target_compile_features(cudnn_full_conv2d PUBLIC cxx_std_11)
