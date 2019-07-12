# ##############################################################################
# Libraries
# ##############################################################################
add_library(Tensor SHARED Tensor.cpp.cu)
add_library(Utils SHARED Utils.cpp)
add_library(cudnnConv2d SHARED cudnnConv2d.cpp.cu)
add_library(manual SHARED manual.cpp.cu)
add_library(hand SHARED cp4Conv2d.cpp.cu)

# ##############################################################################
# Dependencies
# ##############################################################################
target_link_libraries(cudnnConv2d PUBLIC cudnn PRIVATE Tensor)
target_link_libraries(Utils PUBLIC Tensor)
target_link_libraries(manual PUBLIC Tensor)
target_link_libraries(hand PUBLIC Tensor)

# ##############################################################################
# Tests
# ##############################################################################
include(CTest)
enable_testing()
add_executable(test_main
               test/test_main.cpp.cu
               test/test_cudnnConv2d.cpp.cu
               test/test_Tensor.cpp.cu
               test/test_Conv.cpp.cu
               test/test_Utils.cpp.cu
               )

target_link_libraries(test_main
                      PRIVATE doctest
                              manual
                              Utils
                              cudnnConv2d
                              hand
                              Tensor)
add_test(test_main test_main)
