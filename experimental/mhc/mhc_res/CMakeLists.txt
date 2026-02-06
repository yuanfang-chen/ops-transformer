cmake_minimum_required(VERSION 3.16)
project(mhc_res)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(ASCEND_INSTALL_PATH "/usr/local/Ascend/ascend-toolkit/latest" CACHE PATH "Ascend toolkit path")
set(ASCEND_CANN_PACKAGE_PATH ${ASCEND_INSTALL_PATH})
set(SOC_VERSION "ascend910b2" CACHE STRING "SoC version")

set(KERNEL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/kernel)
set(TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/test)

include(${ASCEND_INSTALL_PATH}/compiler/tikcpp/ascendc_kernel_cmake/ascendc.cmake)

ascendc_library(mhc_res_kernel SHARED
    ${KERNEL_DIR}/mhc_res_kernel.cpp
)

macro(add_test_target TARGET_NAME SOURCE_FILE)
    add_executable(${TARGET_NAME} ${SOURCE_FILE})
    target_link_directories(${TARGET_NAME} PRIVATE
        ${ASCEND_INSTALL_PATH}/lib64
    )
    target_link_libraries(${TARGET_NAME}
        mhc_res_kernel
        ascendcl
        runtime
    )
    target_include_directories(${TARGET_NAME} PRIVATE
        ${ASCEND_INSTALL_PATH}/include
    )
endmacro()

add_test_target(test_multi_dtype ${TEST_DIR}/test_multi_dtype.cpp)

enable_testing()
add_test(NAME multi_dtype COMMAND test_multi_dtype)
