# ##############################################################################
# Libraries
# ##############################################################################
add_library(Tensor SHARED Tensor.cu)
add_library(Utils SHARED Utils.cu)
add_library(NVConv2d SHARED NVConv2d.cu)
add_library(cp4conv SHARED cp4Conv2d.cu)
add_library(cp4convForward OBJECT cp4Conv2dForward.cu)
add_library(cp4convBackwardData OBJECT cp4Conv2dBackwardData.cu)
add_library(cp4convBackwardFilter OBJECT cp4Conv2dBackwardFilter.cu)
target_compile_features(Tensor PUBLIC cxx_std_11)

# ##############################################################################
# Dependencies
# ##############################################################################
target_link_libraries(NVConv2d PUBLIC cudnn PRIVATE Tensor)
target_link_libraries(Utils PUBLIC Tensor PRIVATE curand)
target_link_libraries(cp4conv
                      PUBLIC Tensor
                      PRIVATE Utils
                              cp4convForward
                              cp4convBackwardData
                              cp4convBackwardFilter)

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
# add_executable(CP4Conv2dForward cp4Conv2d.cu) add_executable(NVConv2dForward
# NVConv2d.cu)
add_executable(Tensor_Benchmark bench/bench_generic.cu)
add_executable(BenchCP4 bench/BenchCP4.cu)
add_executable(BenchNV bench/BenchNV.cu)

target_compile_features(Tensor_Benchmark PUBLIC cxx_std_11)
target_compile_features(BenchCP4 PUBLIC cxx_std_11)
target_compile_features(BenchNV PUBLIC cxx_std_11)

# target_link_libraries(CP4Conv2dForward PRIVATE Tensor)
# target_link_libraries(NVConv2dForward PRIVATE cudnn Tensor)
target_link_libraries(BenchCP4 PRIVATE cp4conv)
target_link_libraries(BenchNV PRIVATE NVConv2d)

# target_compile_options(CP4Conv2dForward PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
# --generate-line-info >)
