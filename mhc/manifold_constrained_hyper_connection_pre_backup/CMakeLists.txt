# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
# op_host_aclnnInner  如果自己实现了aclnn接口使用这个
# op_host_aclnn  使用自动生成的aclnn接口
# add_ops_compile_options(
#         OP_NAME ManifoldConstrainedHyperConnectionPre
#         OPTIONS --cce-auto-sync=off
#                 -Wno-deprecated-declarations
#                 -Werror
#                 -fpermissive
# )

# set(manifold_constrained_hyper_connection_pre_depends transformer/mhc/manifold_constrained_hyper_connection_pre PARENT_SCOPE)
# target_sources(op_host_aclnnInner PRIVATE
#         op_host/manifold_constrained_hyper_connection_pre_def.cpp
# )

# target_sources(optiling PRIVATE
#         op_host/manifold_constrained_hyper_connection_pre_tiling.cpp
# )

# if (NOT BUILD_OPEN_PROJECT)
#     target_sources(opmaster_ct PRIVATE
#     op_host/manifold_constrained_hyper_connection_pre_tiling.cpp
#     )
# endif ()

# target_include_directories(optiling PRIVATE
#         ${CMAKE_CURRENT_SOURCE_DIR}/op_host
# )

# target_sources(opsproto PRIVATE
#         op_host/manifold_constrained_hyper_connection_pre_infershape.cpp
# )

# target_include_directories(optiling PRIVATE)
# target_sources(opapi PRIVATE
#         op_api/aclnn_manifold_constrained_hyper_connection_pre.cpp
#         op_api/manifold_constrained_hyper_connection_pre.cpp
# )

# install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_manifold_constrained_hyper_connection_pre.h
#         DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
# )

# file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
# if(NOT ENABLE_TEST)
#     list(REMOVE_ITEM CURRENT_DIRS tests)
# endif()
# foreach(SUB_DIR ${CURRENT_DIRS})
#     if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
#         add_subdirectory(${SUB_DIR})
#     endif()
# endforeach()