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
 * \file causal_conv1d_update_apt.cpp
 * \brief CausalConv1dUpdate kernel entry point
 */

#include "causal_conv1d_update.h"

#define TILING_KEY_BASE 30000

extern "C" __global__ __aicore__ void causal_conv1d_update(
    GM_ADDR x,
    GM_ADDR filter,
    GM_ADDR cacheState,
    GM_ADDR cacheIndices,
    GM_ADDR acceptTokenNum,
    GM_ADDR queryStartLoc,
    GM_ADDR y,
    GM_ADDR outputCacheState,
    GM_ADDR tiling)
{
    GET_TILING_DATA(tilingData, tiling);

    // Check if FP16 or BF16
    // Assuming FP16 for now (can be extended to support BF16)
    if (TILING_KEY_IS(TILING_KEY_BASE)) {
        CausalConv1dUpdateKernel<half> op;
        op.Init(x, weight, cacheState, cacheIndices, acceptTokenNum, queryStartLoc,
               y, outputCacheState,
               reinterpret_cast<CausalConv1dUpdateTilingData*>(tilingData.GetDataPtr()));
        op.Process();
    }
}
