/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file aot_tiling_config.h
 * \brief AOT Tiling 配置定义 - 预编译的高频使用配置
 */

#ifndef AOT_TILING_CONFIG_H
#define AOT_TILING_CONFIG_H

#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"
#include "aot_framework.h"

namespace AOT {

// 常用配置1: float16, 不量化, 无smoothScale, 不使用fullmesh_v2
static constexpr uint8_t AOT_MoeDistributeDispatchV2_fp16_unquant_Value[] = {
    8, 0, 0, 0,        // epWorldSize
    0, 0, 0, 0,        // epRankId
    0, 0, 0, 0,        // expertShardType
    0, 0, 0, 0,        // sharedExpertNum
    0, 0, 0, 0,        // sharedExpertRankNum
    64, 0, 0, 0,       // moeExpertNum
    0, 0, 0, 0,        // quantMode
    128, 0, 0, 0,      // globalBs
    16, 0, 0, 0,       // bs
    8, 0, 0, 0,        // k
    0, 16, 0, 0,       // h (4096)
    128, 0, 0, 0,      // a
    8, 0, 0, 0,        // aivNum
    0, 0, 0, 0,        // isTokenMask
    0, 0, 0, 0,        // isExpertMask
    0, 0, 0, 0,        // isPerformance
    0, 0, 0, 0,        // isQuant
    0,                 // reserved0
    0,                 // reserved1
    0,                 // reserved2
    0, 0, 0, 0,        // padding
    254, 2, 0, 0,     // totalUbSize (190 * 1024)
    0, 0, 0, 4,       // totalWinSizeEp (64 * 1024 * 1024)
    0, 0, 0, 0,        // expertTokenNumsType
    0, 0, 0, 0,        // zeroComputeExpertNum
    0, 0, 0, 0,        // cumSumUBMinValue
    0, 0, 0, 0,        // scalesRow
    0, 0, 0, 0,        // scalesCol
    0, 0, 0, 0,        // scalesTypeSize
    0, 0, 0, 0         // scalesCount
};
using AOT_MoeDistributeDispatchV2_fp16_unquant_Type = AOTHolder<MoeDistributeDispatchV2Info, AOT_MoeDistributeDispatchV2_fp16_unquant_Value>;

// 常用配置2: float16, 动态量化, 无smoothScale, 不使用fullmesh_v2
static constexpr uint8_t AOT_MoeDistributeDispatchV2_fp16_dynamic_quant_Value[] = {
    8, 0, 0, 0,               // epWorldSize
    0, 0, 0, 0,            // epRankId
    0, 0, 0, 0,            // expertShardType
    0, 0, 0, 0,            // sharedExpertNum
    0, 0, 0, 0,            // sharedExpertRankNum
    64, 0, 0, 0,           // moeExpertNum
    2, 0, 0, 0,            // quantMode (PERTOKEN_DYNAMIC_QUANT)
    128, 0, 0, 0,          // globalBs
    16, 0, 0, 0,           // bs
    8, 0, 0, 0,            // k
    0, 16, 0, 0,           // h (4096)
    128, 0, 0, 0,          // a
    8, 0, 0, 0,            // aivNum
    0, 0, 0, 0,            // isTokenMask
    0, 0, 0, 0,            // isExpertMask
    0, 0, 0, 0,            // isPerformance
    1, 0, 0, 0,            // isQuant
    0,                     // reserved0
    0,                     // reserved1
    0,                     // reserved2
    0, 0, 0, 0,            // padding
    254, 2, 0, 0,          // totalUbSize (190 * 1024)
    0, 0, 0, 4,            // totalWinSizeEp (64 * 1024 * 1024)
    0, 0, 0, 0,            // expertTokenNumsType
    0, 0, 0, 0,            // zeroComputeExpertNum
    0, 0, 0, 0,            // cumSumUBMinValue
    0, 0, 0, 0,            // scalesRow
    0, 0, 0, 0,            // scalesCol
    0, 0, 0, 0,            // scalesTypeSize
    0, 0, 0, 0             // scalesCount
};
using AOT_MoeDistributeDispatchV2_fp16_dynamic_quant_Type = AOTHolder<MoeDistributeDispatchV2Info, AOT_MoeDistributeDispatchV2_fp16_dynamic_quant_Value>;

// 常用配置3: float16, 不量化, 无smoothScale, 使用fullmesh_v2
static constexpr uint8_t AOT_MoeDistributeDispatchV2_fp16_unquant_fullmesh_Value[] = {
    8, 0, 0, 0,               // epWorldSize
    0, 0, 0, 0,            // epRankId
    0, 0, 0, 0,            // expertShardType
    0, 0, 0, 0,            // sharedExpertNum
    0, 0, 0, 0,            // sharedExpertRankNum
    64, 0, 0, 0,           // moeExpertNum
    0, 0, 0, 0,            // quantMode
    128, 0, 0, 0,          // globalBs
    16, 0, 0, 0,           // bs
    8, 0, 0, 0,            // k
    0, 16, 0, 0,           // h (4096)
    128, 0, 0, 0,          // a
    8, 0, 0, 0,            // aivNum
    0, 0, 0, 0,            // isTokenMask
    0, 0, 0, 0,            // isExpertMask
    0, 0, 0, 0,            // isPerformance
    0, 0, 0, 0,            // isQuant
    0,                     // reserved0
    0,                     // reserved1
    0,                     // reserved2
    0, 0, 0, 0,            // padding
    254, 2, 0, 0,          // totalUbSize (190 * 1024)
    0, 0, 0, 4,            // totalWinSizeEp (64 * 1024 * 1024)
    0, 0, 0, 0,            // expertTokenNumsType
    0, 0, 0, 0,            // zeroComputeExpertNum
    0, 0, 0, 0,            // cumSumUBMinValue
    0, 0, 0, 0,            // scalesRow
    0, 0, 0, 0,            // scalesCol
    0, 0, 0, 0,            // scalesTypeSize
    0, 0, 0, 0             // scalesCount
};
using AOT_MoeDistributeDispatchV2_fp16_unquant_fullmesh_Type = AOTHolder<MoeDistributeDispatchV2Info, AOT_MoeDistributeDispatchV2_fp16_unquant_fullmesh_Value>;

// AOT 注册表
using MoeDistributeDispatchV2AOTRegistry = AOTRegistry<
    AOT_MoeDistributeDispatchV2_fp16_unquant_Type,
    AOT_MoeDistributeDispatchV2_fp16_dynamic_quant_Type,
    AOT_MoeDistributeDispatchV2_fp16_unquant_fullmesh_Type
>;

} // namespace AOT

#endif // AOT_TILING_CONFIG_H
