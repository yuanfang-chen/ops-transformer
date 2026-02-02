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
 * \file moe_distribute_dispatch_v3.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
#include "moe_distribute_dispatch_v3.h"
#include "moe_distribute_dispatch_v3_tiling.h"
#include "moe_distribute_dispatch_v3_full_mesh.h"
#include "moe_distribute_dispatch_v3_tiling_key.h"

#include "moe_distribute_dispatch_a2.h"
#include "moe_distribute_dispatch_a2_layered.h"
#include "moe_distribute_dispatch_a2_layered_aicpu.h"

using namespace MoeDistributeDispatchA2Impl;
using namespace MoeDistributeDispatchV3Impl;
using namespace MoeDistributeDispatchV3FullMeshImpl;
using namespace Mc2Tiling;
using namespace AscendC;

template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch_v3(
    GM_ADDR context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    return;
}