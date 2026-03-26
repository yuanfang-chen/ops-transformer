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
 * \file moe_gating_top_k_softmax_310p.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "moe_gating_top_k_softmax_310p.h"

using namespace AscendC;
using namespace MoeGatingTopKSoftmax;

#define TILINGKEY_WITHOUT_FINISHED_NEED_PAD_ENGINF_310P  18


#define MOE_GATING_TOP_K_SOFTMAX_310P_IMPL(INPUT_TYPE, BUFFER_NUM)                            \
    do {                                                                                               \
        GET_TILING_DATA_WITH_STRUCT(MoeGatingTopKSoftmax310PTilingData, tiling_data_in, tiling);       \
        const MoeGatingTopKSoftmax310PTilingData* __restrict tilingData = &tiling_data_in;             \
        AscendC::TPipe pipe;                                                                           \
        MoeGatingTopKSoftmax310P<half, int32_t> kernel;                                                \
        kernel.Init(x, y, expertIdx, workspace, tilingData, &pipe);                                    \
        kernel.Process();                                                                              \
    } while (0)

extern "C" __global__ __aicore__ void moe_gating_top_k_softmax(GM_ADDR x,
                                                               GM_ADDR finished,
                                                               GM_ADDR y,
                                                               GM_ADDR expertIdx,
                                                               GM_ADDR rowIdx,
                                                               GM_ADDR workspace,
                                                               GM_ADDR tiling)
{
    if (TILING_KEY_IS(18)) {
        MOE_GATING_TOP_K_SOFTMAX_310P_IMPL(half, int32_t);
    }
    return;
}