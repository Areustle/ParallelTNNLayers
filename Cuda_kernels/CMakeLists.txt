###############################################################################
# Libraries
###############################################################################
add_library(Tensor SHARED src/Tensor.cpp.cu)
add_library(cudnn_full_conv2d SHARED src/cudnn_full_conv2d.cpp.cu)

###############################################################################
# Target Properties
###############################################################################

# target_include_directories
target_include_directories(Tensor PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

target_include_directories(cudnn_full_conv2d PUBLIC 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>)

# # target_link_libraries
# target_link_libraries(Tensor PRIVATE)
target_link_libraries(cudnn_full_conv2d PUBLIC cudnn)

###############################################################################
# Tests
###############################################################################

add_executable(test_main
  test/test_main.cpp.cu
  test/test_cudnn_full_conv2d.cpp.cu
  test/test_Tensor.cpp.cu
  )

target_link_libraries(test_main PRIVATE 
  Tensor 
  cudnn_full_conv2d
  doctest
  )

include(CTest)
include(Catch)
catch_discover_tests( test_main )



# target_compile_features(test_cudnn_full_conv2d PRIVATE cxx_std_14)

# add_executable(manual manual.cu) target_compile_features(manual PRIVATE
# cxx_std_14)
