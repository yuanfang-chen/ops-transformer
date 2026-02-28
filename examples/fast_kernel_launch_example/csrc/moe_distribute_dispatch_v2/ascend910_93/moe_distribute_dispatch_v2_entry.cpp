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
 * \file moe_distribute_dispatch_v2.cpp
 * \brief
 */
#if ASC_DEVKIT_MAJOR >= 9
#include "basic_api/kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#endif
#include "op_kernel/moe_distribute_dispatch_v2.h"
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"

// #if defined(__DAV_C310__)
// using namespace MoeDistributeDispatchA5Impl;
// #else
// using namespace MoeDistributeDispatchA2Impl;
// #endif

using namespace MoeDistributeDispatchV2Impl;
using namespace Mc2Kernel;
// using namespace MoeDistributeDispatchV2FullMeshImpl;
// using namespace Mc2Tiling;
using namespace AscendC;

template<typename XType, typename ExpandxType, int32_t QuantMode, bool IsSmoothScaleExist, bool IsNeedAllgather>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingData)
{
// REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);

    TPipe pipe;
    MoeDistributeDispatchV2<XType, ExpandxType, QuantMode, IsSmoothScaleExist, IsNeedAllgather> op;
    op.InitWithMc2Context(x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, 
        tpSendCountsOut, workspaceGM, mc2Context, tilingData, &pipe);
        
    op.Process();
    return;
}


extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2_Generic(
    int8_t type,
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingData)
{
    // 根据不同的数据类型调用不同的模板
    switch (type) {
        case 1:
            moe_distribute_dispatch_v2<float16_t, float16_t, MoeDistributeDispatchV2Impl::UNQUANT, false, false>(
                x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, 
                assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData);
            break;
        default:
            AscendC::PRINTF("moe_distribute_dispatch_v2 Error: invalid type = %d\n", type);
            return;
    }
}


// <<<>>>调用函数
void moe_distribute_dispatch_v2_demo(int8_t type, uint32_t blockDim, void* stream, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales,
    GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingData)
{
    moe_distribute_dispatch_v2_Generic<<<blockDim, nullptr, stream>>>(
        type, 
        x, expertIds, scales, xActiveMask, expertScales, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, tpSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingData
    );
}