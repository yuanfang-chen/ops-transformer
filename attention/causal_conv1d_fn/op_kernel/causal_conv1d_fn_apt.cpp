/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file causal_conv1d_fn_apt.cpp
 * \brief CausalConv1dFn 核函数入口。
 *
 */

#include "arch35/causal_conv1d_fn.h"

using namespace CausalConv1dFnNs;

// ============================================================================
// TilingKey 枚举
// 用于在编译期（constexpr if）分派不同数据类型的 kernel 实例
// ============================================================================
#define TILING_KEY_FN_BF16 10000
#define TILING_KEY_FN_FP16 10001
// ============================================================================
// 核函数入口
//
// 模板参数 schMode 由 tiling 侧通过 SetTilingKey 写入，
// 在 kernel 侧通过 if constexpr 选择对应数据类型的实现。
// ============================================================================
__global__ __aicore__ void causal_conv1d_fn(
    GM_ADDR x,                    // 输入1: x
    GM_ADDR weight,               // 输入2: weight
    GM_ADDR convStates,           // 输入3: convStates (cache states)
    GM_ADDR queryStartLoc,        // 输入4: queryStartLoc (OPTIONAL)
    GM_ADDR cacheIndices,         // 输入5: cacheIndices (OPTIONAL)
    GM_ADDR initialStateMode,     // 输入6: initialStateMode (OPTIONAL)
    GM_ADDR bias,                 // 输入7: bias (OPTIONAL, 暂不使用)
    GM_ADDR numAcceptedToken,     // 输入8: numAcceptedToken (OPTIONAL, 暂不使用)
    GM_ADDR y,                    // 输出1: y
    GM_ADDR convStatesOut,        // 输出2: convStates (与输入3为同一地址，inplace)
    GM_ADDR workspace,            // workspace
    GM_ADDR tiling)               // tiling data
{
    REGISTER_TILING_DEFAULT(CausalConv1dFnTilingData);
    GET_TILING_DATA_WITH_STRUCT(CausalConv1dFnTilingData, tilingData, tiling);

    if (TILING_KEY_IS(TILING_KEY_FN_BF16)) {
        CausalConv1dFn<bfloat16_t> op;
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices,
                initialStateMode, y, workspace, &tilingData);
        op.Process();
    }

    if (TILING_KEY_IS(TILING_KEY_FN_FP16))  {
        CausalConv1dFn<half> op;
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices,
                initialStateMode, y, workspace, &tilingData);
        op.Process();
    }
}