/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_v2_entry.cpp
 * \brief Kernel entry implementation for direct launch (<<<>>> syntax)
 */

#include "moe_distribute_combine_v2_entry.h"
#include "op_kernel/moe_distribute_combine_v2_tiling.h"

#include "kernel_operator.h"
#include "moe_distribute_combine_v2.h"

using namespace AscendC;
using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Kernel;

constexpr uint32_t TILINGKEY_NO_QUANT = 0;
constexpr uint32_t TILINGKEY_INT8_QUANT = 2;
constexpr uint32_t TILINGKEY_TPL_MTE = 0;
constexpr uint32_t TILINGKEY_TPL_HIERARCHY = 2;

template <bool HasTp, uint8_t QuantMode>
__global__ __aicore__ void
moe_distribute_combine_v2_kernel(GM_ADDR mc2Context, GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                 GM_ADDR epSendCount, GM_ADDR tpSendCount, GM_ADDR scales, GM_ADDR xActiveMask,
                                 GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR performanceInfo,
                                 GM_ADDR xOut, GM_ADDR workspaceGM, MoeDistributeCombineV2TilingData tilingData)
{
    TPipe pipe;

    if constexpr (QuantMode == TILINGKEY_NO_QUANT) {
        MoeDistributeCombineV2<bfloat16, bfloat16, int32_t, HasTp, TILINGKEY_NO_QUANT, false> op;
        op.Init(mc2Context, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, nullptr, nullptr,
                nullptr, xActiveMask, sharedExpertX, elasticInfo, oriX, nullptr, nullptr, nullptr, performanceInfo,
                nullptr, nullptr, xOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if constexpr (QuantMode == TILINGKEY_INT8_QUANT) {
        MoeDistributeCombineV2<int8_t, bfloat16, int32_t, HasTp, TILINGKEY_INT8_QUANT, false> op;
        op.Init(mc2Context, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, nullptr, nullptr,
                nullptr, xActiveMask, sharedExpertX, elasticInfo, oriX, nullptr, nullptr, nullptr, performanceInfo,
                nullptr, nullptr, xOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}

void moe_distribute_combine_v2_entry(int32_t tilingKey, uint32_t blockDim, aclrtStream stream, void *mc2Context,
                                     void *expandX, void *expertIds, void *assistInfoForCombine, void *epSendCount,
                                     void *tpSendCount, void *scales, void *xActiveMask, void *sharedExpertX,
                                     void *elasticInfo, void *oriX, void *performanceInfo, void *xOut, void *workspace,
                                     const void *tilingData)
{
    const MoeDistributeCombineV2TilingData *tiling = static_cast<const MoeDistributeCombineV2TilingData *>(tilingData);

    moe_distribute_combine_v2_kernel<false, 0><<<blockDim, nullptr, stream>>>(
        (GM_ADDR)mc2Context, (GM_ADDR)expandX, (GM_ADDR)expertIds, (GM_ADDR)assistInfoForCombine, (GM_ADDR)epSendCount,
        (GM_ADDR)tpSendCount, (GM_ADDR)scales, (GM_ADDR)xActiveMask, (GM_ADDR)sharedExpertX, (GM_ADDR)elasticInfo,
        (GM_ADDR)oriX, (GM_ADDR)performanceInfo, (GM_ADDR)xOut, (GM_ADDR)workspace, *tiling);
}