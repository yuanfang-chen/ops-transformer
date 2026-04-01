# ---------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under terms and conditions of 
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to License for details. You may not use this file except in compliance with License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. 
# See LICENSE in root of software repository for full text of the License.
# ---------------------------------------------------------------------------------------------------------
message(STATUS "=== Debug: start ops.index.spatial_transformer_op_kernel_aicpu!!!!!!!!.op_kernel_aicpu.CMakeLists.txt ")

set(ASCEND_OP_NAME "" CACHE STRING "Ascend op names to compile")
set(OP_TYPE "spatial_transformer")

skip_aicpu_kernel("${OP_TYPE}" "${ASCEND_OP_NAME}")
if(SKIP_AICPU_FLAG)
  return()
endif()


if (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
  message(STATUS "=== spatial_transformer_op_kernel_aicpu!!!!!!!!.case1 ")
  if (NOT (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
    add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
    set(CMAKE_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
  endif()
  message(STATUS "=== spatial_transformer_op_kernel_aicpu!!!!!!!!.case1.1 ")
  # aicpu json
  message(STATUS "=== ${CMAKE_CURRENT_SOURCE_DIR}")
  file(GLOB_RECURSE JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
  message(STATUS "=== spatial_transformer_op_kernel_aicpu!!!!!!!!.case1.2 ")
  set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES ${JSON_FILE})
  message(STATUS "=== spatial_transformer_op_kernel_aicpu!!!!!!!!.case1.3 ")
  # aicpu cust kernel
  file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu*.cpp)
  message(STATUS "[spatial_transformer] Found aicpu sources: ${AICPU_SRC}, ascend dir: ${ASCEND_DIR}, ophsot name: ${OPHOST_NAME}")

  set(OBJ_NAME spatial_transformer_cust_obj)
  add_aicpu_cust_kernel_modules(${OBJ_NAME})
  target_sources(${OBJ_NAME} PRIVATE ${AICPU_SRC})
else()
  message(STATUS "=== spatial_transformer_op_kernel_aicpu!!!!!!!!.case2 ")
  add_modules_sources(HOSTNAME ${OPHOST_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR} OPTYPE spatial_transformer ACLNNTYPE aclnn_exclude)
endif()

 if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
     AddAicpuOpTestCase(spatial_transformer)
  endif()

message(STATUS "=== end: start ops.index.spatial_transformer_op_kernel_aicpu!!!!!!!!.op_kernel_aicpu.CMakeLists.txt ")