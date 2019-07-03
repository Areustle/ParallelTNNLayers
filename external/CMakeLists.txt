# Prepare doctest for other targets to use
add_library(doctest INTERFACE IMPORTED GLOBAL)
target_include_directories(doctest INTERFACE 
    $<INSTALL_INTERFACE:include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>/doctest)

