# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(moe_dispatch)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOC_VERSION "Ascend910_9382" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH}
    CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

file(GLOB KERNEL_FILES 
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_dispatch.cpp
)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist ,please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)
set(SHMEM_HOME_PATH  $ENV{SHMEM_HOME_PATH})

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(ascendc_kernels SHARED ${KERNEL_FILES})

ascendc_include_directories(ascendc_kernels SHARED
${SHMEM_HOME_PATH}/shmem/include
${CMAKE_CURRENT_SOURCE_DIR}
${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
${ASCEND_CANN_PACKAGE_PATH}/include
)

add_executable(test_moe_dispatch 
    ${CMAKE_CURRENT_SOURCE_DIR}/test_moe_dispatch.cpp
)

target_compile_options(test_moe_dispatch PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

# target_compile_definitions(test_moe_dispatch PRIVATE
#     SOC_VERSION="${SOC_VERSION}"
# )

target_include_directories(test_moe_dispatch PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${SHMEM_HOME_PATH}/shmem/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/aclnn
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_directories(test_moe_dispatch PRIVATE
    ${SHMEM_HOME_PATH}/shmem/lib

)

target_link_libraries(test_moe_dispatch PRIVATE
    host_intf_pub
    ascendc_kernels
    tiling_api
    platform
    pthread
    hccl
    hcomm
    shmem
    nnppbase
)

install(TARGETS test_moe_dispatch
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)