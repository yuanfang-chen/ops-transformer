# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_sources("--npu-arch=dav-2201")
if (BUILD_TORCH_OPS)
    set(OPERATOR_NAME "MoeDistributeDispatchV2")
    message(STATUS "BUILD_TORCH_OPS ON in ${OPERATOR_NAME}")

    set(OPERATOR_TARGET "${OPERATOR_NAME}_objects")
    set(OPERATOR_CONFIG "${OPERATOR_NAME}:${OPERATOR_TARGET}" PARENT_SCOPE)

    # 包含所有.cpp文件
    file(GLOB OPERATOR_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_dispatch_v2_entry.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/update_context.cpp"
        "${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_dispatch_v2_torch.cpp"
    )
    # Mark .cpp files with special properties
    set_source_files_properties(
        ${OPERATOR_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "--cce-soc-version=Ascend910_93 --cce-soc-core-type=CubeCore  --cce-aicore-input-parameter-size=4096 --cce-auto-sync -xcce"
        # COMPILE_FLAGS "--npu-arch=dav-2201 -xasc"

    )

    target_include_directories(${OPERATOR_TARGET} PRIVATE 
        ${COMMON_INCLUDE_DIRS}
    )
endif()
