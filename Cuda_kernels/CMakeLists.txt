# ##############################################################################
# Libraries
# ##############################################################################
add_library(Tensor SHARED Tensor.cu)
add_library(Utils SHARED Utils.cu)
add_library(NVConv2d SHARED NVConv2d.cu)
add_library(cp4conv SHARED cp4Conv2d.cu)
target_compile_features(Tensor PUBLIC cxx_std_11)

# ##############################################################################
# Dependencies
# ##############################################################################
target_link_libraries(NVConv2d PUBLIC cudnn PRIVATE Tensor)
target_link_libraries(Utils PUBLIC Tensor)
target_link_libraries(cp4conv PUBLIC Tensor)

# ##############################################################################
# Tests
# ##############################################################################
include(CTest)
enable_testing()
add_executable(test_main
               test/test_main.cpp.cu
               test/test_NVConv2d.cpp.cu
               test/test_Tensor.cpp.cu
               test/test_cp4Conv.cpp.cu
               test/test_Utils.cpp.cu)

target_link_libraries(test_main
                      PRIVATE doctest
                              Utils
                              NVConv2d
                              cp4conv
                              Tensor)

add_test(test_main test_main)

# ##############################################################################
# Benchmarks
# ##############################################################################
add_executable(CP4Conv2dForward cp4Conv2d.cu)
add_executable(NVConv2dForward NVConv2d.cu)
add_executable(Tensor_Benchmark bench/bench_generic.cu)
target_compile_features(Tensor_Benchmark PUBLIC cxx_std_11)

target_link_libraries(CP4Conv2dForward PRIVATE Tensor)
target_link_libraries(NVConv2dForward PRIVATE cudnn Tensor)
