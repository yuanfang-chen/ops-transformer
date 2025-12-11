# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################
set(_FusedFloydAttentionGrad_OpProtoSourcesExt
        ${CMAKE_SOURCE_DIR}/attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_infershape.cpp
)

set(_FusedFloydAttentionGrad_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling_common.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling_s1s2_bn2gs1s2.cpp
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling.cpp
)

set(_FusedFloydAttentionGrad_KernelTilingDataDefH
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling_common.h
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling_s1s2_bn2gs1s2_def.h
        ${OPS_TRANSFORMER_DIR}/attention/fused_floyd_attention_grad/op_host/fused_floyd_attention_grad_tiling.h
)

set(_FusedFloydAttentionGrad_TargetPrivateLinkLibrariesExt)

set(_FusedFloydAttentionGrad_TilingPrivateIncludesExt
        ${CMAKE_SOURCE_DIR}/attention/fused_floyd_attention_grad/op_host/
        ${CMAKE_SOURCE_DIR}/attention/fused_floyd_attention_grad/op_host/op_api/
)

set(_FusedFloydAttentionGrad_UTestCommonPrivateIncludeExt
        ${OPBASE_INC_DIRS}
        ${OPS_TRANSFORMER_DIR}/common/include
)

set(__FusedFloydAttentionGrad_COMPRESS_GRAD_TargetPrivateLinkLibrariesExt
        ${UTest_NamePrefix}_OpTiling
        error_manager
)

aux_source_directory(${CMAKE_SOURCE_DIR}/attention/fused_floyd_attention_grad/op_kernel _FusedFloydAttentionGrad_KernelSourcesExt)

OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer/attention
        BRIEF                                   FusedFloydAttentionGrad
        SNAKE                                   fused_floyd_attention_grad
        PROTO_SOURCES_EXT                       ${_FusedFloydAttentionGrad_OpProtoSourcesExt}
        TILING_SOURCES_EXT                      ${_FusedFloydAttentionGrad_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_FusedFloydAttentionGrad_TilingPrivateIncludesExt}
        KERNEL_SOURCES_EXT                      ${_FusedFloydAttentionGrad_KernelSourcesExt}
        KERNEL_TILING_DATA_DEF_H                ${_FusedFloydAttentionGrad_KernelTilingDataDefH}
        UTEST_COMMON_PRIVATE_LINK_LIBRARIES_EXT ${_FusedFloydAttentionGrad_TargetPrivateLinkLibrariesExt}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_FusedFloydAttentionGrad_UTestCommonPrivateIncludeExt}
)
