# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
# op_host_aclnnInner  如果自己实现了aclnn接口使用这个
# op_host_aclnn  使用自动生成的aclnn接口
add_ops_compile_options(
        OP_NAME MatrixTransformRope
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -fpermissive
)

set(matrix_transform_rope_depends your_category/matrix_transform_rope PARENT_SCOPE)

target_sources(op_host_aclnnInner PRIVATE
        op_host/matrix_transform_rope_def.cpp
)

target_sources(optiling PRIVATE
        op_host/matrix_transform_rope_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
    op_host/matrix_transform_rope_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

target_sources(opsproto PRIVATE
        op_host/matrix_transform_rope_infershape.cpp
)

target_include_directories(optiling PRIVATE)
target_sources(opapi PRIVATE
        op_api/aclnn_matrix_transform_rope.cpp
        op_api/aclnn_matrix_transform_rope_process.cpp
        op_api/matrix_transform_rope.cpp
)

install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_matrix_transform_rope.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

if (NOT BUILD_OPS_RTY_KERNEL)
    file(GLOB OPAPI_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_api/*.cpp)
    message(STATUS "=== Debug: OPAPI_SRCS =${OPAPI_SRCS}")
    if (OPAPI_SRCS)
        add_opapi_modules()
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
    endif()
endif()
