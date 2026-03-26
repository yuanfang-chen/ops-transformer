# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------


cmake_minimum_required(VERSION 3.16)
find_package(ASC REQUIRED)
project(kernel_samples LANGUAGES ASC CXX)
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
)
find_package(Torch REQUIRED HINTS ${TORCH_CMAKE_PREFIX_PATH})

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import os, torch_npu; print(os.path.dirname(torch_npu.__file__))"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE TORCH_NPU_PATH
)
set(TORCH_NPU_INCLUDE_DIRS 
    ${TORCH_NPU_PATH}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/examples
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/block
    ${CMAKE_CURRENT_SOURCE_DIR}/include/kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/include/memcopy
    ${CMAKE_CURRENT_SOURCE_DIR}/include/utils
    ${CMAKE_CURRENT_SOURCE_DIR}/include/vf
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/basic_api/impl
    ${ASCEND_HOME}/${CMAKE_SYSTEM_PROCESSOR}-linux/ascendc/include/basic_api
    ${ASCEND_HOME}/include/ascendc
    ${ASCEND_HOME}/include/ascendc/basic_api/interface
    ${ASCEND_HOME}/include/ascendc/basic_api
    ${ASCEND_HOME}/pkg_inc/op_common
    ${ASCEND_HOME}/pkg_inc/base
    ${ASCEND_HOME}/pkg_inc
)
set(TORCH_NPU_LIBRARIES ${TORCH_NPU_PATH}/lib)

execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import pybind11; print(pybind11.get_cmake_dir())"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE PYBIND11_CMAKE_PREFIX_PATH
)
find_package(pybind11 REQUIRED HINTS ${PYBIND11_CMAKE_PREFIX_PATH})

set_source_files_properties(
    fused_infer_attention_score.cpp
    PROPERTIES LANGUAGE ASC
)

set_source_files_properties(
    include/op_host/abc.cpp
    PROPERTIES LANGUAGE CXX
)

add_library(xxxlib SHARED
    include/op_host/abc.cpp
)


pybind11_add_module(ascendc_ops SHARED
    fused_infer_attention_score.cpp
    # abc.cpp
)

target_link_libraries(ascendc_ops PRIVATE
    xxxlib
)

target_include_directories(ascendc_ops PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../../..
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${COMMON_INCLUDE_DIRS}
)

target_link_libraries(ascendc_ops PRIVATE
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
)

target_link_directories(ascendc_ops PRIVATE
    ${TORCH_NPU_LIBRARIES}
    ${ASCEND_HOME}/lib64
)

# ======================================================================================
# NPU 编译选项配置
#
# 说明：
#   - 需根据实际部署的 NPU 硬件架构选择对应的 `npu-arch` 参数。
# ======================================================================================
target_compile_options(ascendc_ops PRIVATE
    ${TORCH_CXX_FLAGS}
    # $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-2201>
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3101>
    "-xasc"
    "-w"
    "-O3"
    # "--asc-aicore-lang"
)

target_compile_options(xxxlib PRIVATE
    ${TORCH_CXX_FLAGS}
    "--save-temps"
    "-w"
    "-O3"
)