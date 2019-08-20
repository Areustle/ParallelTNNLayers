# ##############################################################################
# Libraries
# ##############################################################################
add_library(Tensor SHARED Tensor.cu)
add_library(Utils SHARED Utils.cu)
add_library(NVConv2d SHARED NVConv2d.cu)
# add_library(cp4conv SHARED cp4Conv2d.cu)
target_compile_features(Tensor PUBLIC cxx_std_11)

# ##############################################################################
# Dependencies
# ##############################################################################
target_link_libraries(NVConv2d PUBLIC cudnn PRIVATE Tensor)
target_link_libraries(Utils PUBLIC Tensor PRIVATE curand)
# target_link_libraries(cp4conv PUBLIC Tensor)

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
                              Tensor)

add_test(test_main test_main)

# ##############################################################################
# Benchmarks
# ##############################################################################
# add_executable(BenchCP4BatchSize bench/BenchCP4BatchSize.cu)
# add_executable(BenchCP4ChannelDepth bench/BenchCP4ChannelDepth.cu)
# add_executable(BenchCP4FilterCount bench/BenchCP4FilterCount.cu)
# add_executable(BenchCP4FilterSize bench/BenchCP4FilterSize.cu)
# add_executable(BenchCP4ImageSize bench/BenchCP4ImageSize.cu)
# add_executable(BenchCP4SingleTensor bench/BenchCP4SingleTensor.cu)
# add_executable(BenchNV bench/BenchNV.cu)

# target_compile_features(BenchCP4BatchSize PUBLIC cxx_std_11)
# target_compile_features(BenchCP4ChannelDepth PUBLIC cxx_std_11)
# target_compile_features(BenchCP4FilterCount PUBLIC cxx_std_11)
# target_compile_features(BenchCP4FilterSize PUBLIC cxx_std_11)
# target_compile_features(BenchCP4ImageSize PUBLIC cxx_std_11)
# target_compile_features(BenchCP4SingleTensor PUBLIC cxx_std_11)
# target_compile_features(BenchNV PUBLIC cxx_std_11)

# # target_link_libraries(BenchCP4 PRIVATE cp4conv)
# target_link_libraries(BenchNV PRIVATE NVConv2d)
