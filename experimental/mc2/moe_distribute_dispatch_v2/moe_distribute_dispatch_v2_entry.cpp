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
#include "./kernel/moe_distribute_dispatch_v2.h"
#include "./kernel/moe_distribute_dispatch_v2_tiling.h"
#include "./kernel/moe_distribute_dispatch_v2_full_mesh.h"
#include "./kernel/moe_distribute_dispatch_v2_layered.h"
#include "./kernel/moe_distribute_dispatch_v2_tiling_key.h"

#if defined(__DAV_C310__)
#include "arch35/moe_distribute_dispatch_arch35.h"
#else
#include "./kernel/moe_distribute_dispatch_a2.h"
#include "./kernel/moe_distribute_dispatch_a2_layered.h"
#include "./kernel/moe_distribute_dispatch_a2_layered_aicpu.h"
#endif // defined(__DAV_C310__)

#if defined(__DAV_C310__)
using namespace MoeDistributeDispatchA5Impl;
#else
using namespace MoeDistributeDispatchA2Impl;
#endif

using namespace MoeDistributeDispatchV2Impl;
using namespace Mc2Kernel;
using namespace MoeDistributeDispatchV2FullMeshImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__attribute__((always_inline)) __aicore__ __inline__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
{
// REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);

    TPipe pipe;
    MoeDistributeDispatchV2Layered<DTYPE_X, DTYPE_EXPAND_X, false, true, false> op;
    op.Init(x, expertIds, scales, xActiveMask, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
        epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingGM, &pipe);
    op.Process();
}


extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2_Generic(
    int8_t type,
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR mc2Context, GM_ADDR tilingGM)
{
    // 根据不同的数据类型调用不同的模板
    switch (type) {
        case 1:
            moe_distribute_dispatch_v2<float16_t, float16_t, MoeDistributeDispatchV2Impl::UNQUANT, false, false>(
                x, scales, output, workspaceGM, mc2Context, tilingGM);
            break;
        default:
            AscendC::PRINTF("moe_distribute_dispatch_v2 Error: invalid type = %d\n", type);
            return;
    }
}


// <<<>>>调用函数
void moe_distribute_dispatch_v2_demo(int8_t type, uint32_t blockDim, void* stream, uint8_t* x, uint8_t* expertIds, uint8_t* scales,
    uint8_t* xActiveMask, uint8_t* expertScales, uint8_t* expandXOut, uint8_t* dynamicScalesOut, uint8_t* assistInfoOut,
    uint8_t* expertTokenNumsOut, uint8_t* epSendCountsOut, uint8_t* expandScalesOut, uint8_t* workspaceGM, uint8_t* mc2Context, uint8_t* tilingGM) {
    moe_distribute_dispatch_v2_Generic<<<blockDim, nullptr, stream>>>(
        type, 
        x, expertIds, scales, xActiveMask, expertScales, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
                    epSendCountsOut, expandScalesOut, workspaceGM, mc2Context, tilingGM
    );
}