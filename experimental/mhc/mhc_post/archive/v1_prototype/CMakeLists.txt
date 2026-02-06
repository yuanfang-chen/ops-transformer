cmake_minimum_required(VERSION 3.16)
project(mhc_post_v1)

set(CMAKE_CXX_STANDARD 17)
set(ASCEND_PATH "/usr/local/Ascend/ascend-toolkit/latest")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_PATH})
set(SOC_VERSION "ascend910b2" CACHE STRING "SoC version")

include(${ASCEND_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(mhc_post_kernel STATIC kernel/mhc_post_kernel.cpp)

macro(add_test_exe NAME)
    add_executable(${NAME} test/${NAME}.cpp)
    target_link_directories(${NAME} PRIVATE ${ASCEND_PATH}/lib64)
    target_link_libraries(${NAME} mhc_post_kernel ascendcl runtime)
    target_include_directories(${NAME} PRIVATE ${ASCEND_PATH}/include)
endmacro()

add_test_exe(test_mhc_post)
