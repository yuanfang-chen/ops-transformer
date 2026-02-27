/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "kernel/moe_distribute_dispatch_shmem.h"

extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2_generic(GM_ADDR gva, GM_ADDR x, 
        GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
        GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
        GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
        GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    AscendC::TPipe pipe;
    MoeDistributeDispatchShmemImpl::MoeDistributeDispatchShmem<bfloat16_t, bfloat16_t, false, false, false, false> op;
    op.Init(gva, x, expertIds, scales, xActiveMask, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut,
            tpSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingGM);
    op.Process();
    return;
}

void moe_dispatch_demo(uint32_t blockDim, void *stream, uint64_t fftsAddr, GM_ADDR gva, GM_ADDR x, 
        GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
        GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
        GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
        GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    moe_distribute_dispatch_v2_generic<<<blockDim, nullptr, stream>>>(
        gva, x, expertIds, scales, xActiveMask, expertScales, 
        elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, 
        assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, 
        expandScalesOut, workspaceGM, tilingGM
    );
}