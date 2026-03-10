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
 * \file causal_conv1d_apt.cpp
 * \brief CausalConv1dUpdate kernel entry point
 */

#include "./arch35/causal_conv1d_update.h"
#include "./arch35/causal_conv1d_fn.h"

#define TILING_KEY_FN_BF16 10000
#define TILING_KEY_FN_FP16 10001
#define TILING_KEY_UPDATE_FP16 20000

using namespace CausalConv1dFnNs;
extern "C" __global__ __aicore__ void causal_conv1d(
    GM_ADDR x,                    // 输入0: x
    GM_ADDR weight,               // 输入1: weight
    GM_ADDR convStates,           // 输入2: convStates
    GM_ADDR queryStartLoc,        // 输入3: queryStartLoc (optional)
    GM_ADDR cacheIndices,         // 输入4: cacheIndices (optional)
    GM_ADDR initialStateMode,     // 输入5: initialStateMode (optional)
    GM_ADDR bias,                 // 输入6: bias (optional)
    GM_ADDR numAcceptedToken,     // 输入7: numAcceptedToken (optional)
    GM_ADDR y,                    // 输出0: y
    GM_ADDR outputConvStates,     // 输出1: convStates output
    GM_ADDR workspace,            // workspace
    GM_ADDR tiling)               // tiling
{
    REGISTER_TILING_DEFAULT(CausalConv1dUpdateTilingData);
    TPipe pipe;
    // Check if FP16 or BF16
    // Assuming FP16 for now (can be extended to support BF16)
    PRINTF("START UPDATE");
    if (TILING_KEY_IS(TILING_KEY_UPDATE_FP16)) {
        // REGISTER_TILING_DEFAULT(CausalConv1dUpdateTilingData);
        GET_TILING_DATA_WITH_STRUCT(CausalConv1dUpdateTilingData, tilingData, tiling);
        CausalConv1dUpdateKernel<half> op(&pipe);
        PRINTF("START Init");
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices, numAcceptedToken, y, &tilingData);
        PRINTF("START Process");
        op.Process();
        PRINTF("END Process");
    }

    if (TILING_KEY_IS(TILING_KEY_FN_BF16)) {
        // REGISTER_TILING_DEFAULT(CausalConv1dFnTilingData);
        GET_TILING_DATA_WITH_STRUCT(CausalConv1dFnTilingData, tilingData, tiling);
        CausalConv1dFn<bfloat16_t> op;
        PRINTF("START InitA");
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices,
                initialStateMode, y, workspace, &tilingData);
    }

    if (TILING_KEY_IS(TILING_KEY_FN_FP16))  {
        // REGISTER_TILING_DEFAULT(CausalConv1dFnTilingData);
        GET_TILING_DATA_WITH_STRUCT(CausalConv1dFnTilingData, tilingData, tiling);
        CausalConv1dFn<half> op;
        PRINTF("START InitB");
        op.Init(x, weight, convStates, queryStartLoc, cacheIndices,
                initialStateMode, y, workspace, &tilingData);
    }
}
