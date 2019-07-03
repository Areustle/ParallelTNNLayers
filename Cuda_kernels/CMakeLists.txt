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
target_link_libraries(Tensor PRIVATE doctest)
target_link_libraries(cudnn_full_conv2d 
  PUBLIC cudnn
  PRIVATE doctest)

###############################################################################
# Tests
###############################################################################
add_executable(test_main test/test_main.cpp.cu )

target_link_libraries(test_main PRIVATE doctest Tensor cudnn_full_conv2d)



# target_compile_features(test_cudnn_full_conv2d PRIVATE cxx_std_14)

# add_executable(manual manual.cu) target_compile_features(manual PRIVATE
# cxx_std_14)
