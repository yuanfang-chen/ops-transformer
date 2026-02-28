# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
if (BUILD_TORCH_OPS)
    set(OPERATOR_NAME "moedistributedispatchv2")
    message(STATUS "BUILD_TORCH_OPS ON in ${OPERATOR_NAME}")
    
    set(OPERATOR_TARGET "${OPERATOR_NAME}_objects")
    set(OPERATOR_CONFIG "${OPERATOR_NAME}:${OPERATOR_TARGET}" PARENT_SCOPE)

    # 包含所有.cpp文件
    file(GLOB OPERATOR_SOURCES 
        "${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_dispatch_v2_torch.cpp"
    )

    # Mark .cpp files with special properties
    set_source_files_properties(
        ${OPERATOR_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "--cce-soc-version=Ascend910_93 --cce-soc-core-type=CubeCore  --cce-aicore-input-parameter-size=4096 --cce-auto-sync -xcce"
        # COMPILE_FLAGS "--npu-arch=dav-2201 -xasc"
 	
    )

    add_library(${OPERATOR_TARGET} OBJECT ${OPERATOR_SOURCES})

     target_compile_options(${OPERATOR_TARGET} PRIVATE 
        ${COMMON_COMPILE_OPTIONS}
        -DORIG_DTYPE_X=DT_BFLOAT16,DT_FLOAT16,
        -DDTYPE_X=bfloat16_t,float16_t

        -DORIG_DTYPE_EXPERT_IDS=INT
        -DDTYPE_EXPERT_IDS=int

        -DORIG_DTYPE_EXPAND_X=DT_BFLOAT16,DT_FLOAT16,INT
        -DDTYPE_EXPAND_X=bfloat16_t,float16_t,int

        -DORIG_DTYPE_DYNAMIC_SCALES=DT_FLOAT32
        -DDTYPE_DYNAMIC_SCALES=float32_t

        -DORIG_DTYPE_ASSIST_INFO_FORCOMBINE=INT
        -DDTYPE_ASSIST_INFO_FORCOMBINE=int

        -DORIG_DTYPE_EXPERT_TOKEN_NUMS=INT
        -DDTYPE_EXPERT_TOKEN_NUMS=int

        -DORIG_DTYPE_EP_RECV_COUNTS=INT
        -DDTYPE_EP_RECV_COUNTS=int

        -DORIG_DTYPE_TP_RECV_COUNTS=INT
        -DDTYPE_TP_RECV_COUNTS=int

        -DORIG_DTYPE_EXPAND_SCALES=DT_FLOAT32
        -DDTYPE_EXPAND_SCALES=float32_t

        -DDTYPE_SCALES=float32_t
        -DDTYPE_X_ACTIVE_MASK=bool
        -DDTYPE_EXPERT_SCALES=float32_t
        -DDTYPE_ELASTIC_INFO=int
        -DDTYPE_PERFORMANCE_INFO=int
     )

    target_include_directories(${OPERATOR_TARGET} PRIVATE 
        ${COMMON_INCLUDE_DIRS}
    )
endif()