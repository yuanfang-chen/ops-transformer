cmake_minimum_required(VERSION 3.16)
project(flash_attention2_ascendc)
set(CMAKE_CXX_STANDARD 17)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/8.3.RC1/aarch64-linux"
    CACHE STRING "Ascend toolkit path")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/ascend-toolkit/latest")
set(SOC_VERSION "ascend910premiuma" CACHE STRING "SoC version")

include(ExternalProject)
include(${ASCEND_INSTALL_PATH}/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

# --- Device kernel ---
ascendc_library(fa2_kernel STATIC
    ${CMAKE_CURRENT_SOURCE_DIR}/fa2_kernel.cpp
)
target_include_directories(fa2_kernel_interface PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# --- C++ test ---
add_executable(test_fa2_host
    ${CMAKE_CURRENT_SOURCE_DIR}/test_fa2_host.cpp
)
target_include_directories(test_fa2_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_directories(test_fa2_host PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_link_libraries(test_fa2_host
    fa2_kernel ascendcl runtime -Wl,--no-as-needed
)
add_dependencies(test_fa2_host fa2_kernel)

# --- Shared library for Python integration ---
add_library(fa2_ops SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/fa2_wrapper.cpp
)
target_include_directories(fa2_ops PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_directories(fa2_ops PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
)
target_link_libraries(fa2_ops
    fa2_kernel ascendcl runtime -Wl,--no-as-needed
)
add_dependencies(fa2_ops fa2_kernel)
set_target_properties(fa2_ops PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib
)
