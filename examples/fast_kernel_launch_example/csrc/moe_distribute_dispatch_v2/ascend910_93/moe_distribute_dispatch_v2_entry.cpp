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
 * \file moe_distribute_dispatch_v2_entry.cpp
 * \brief Kernel entry implementation for direct launch (<<<>>> syntax)
 */

#include "moe_distribute_dispatch_v2_entry.h"
#include "op_kernel/moe_distribute_dispatch_v2_tiling.h"

#include "kernel_operator.h"
#include "moe_distribute_dispatch_v2.h"
#include "moe_distribute_dispatch_v2_full_mesh.h"

using namespace AscendC;
using namespace MoeDistributeDispatchV2Impl;
using namespace MoeDistributeDispatchV2FullMeshImpl;
using namespace Mc2Kernel;

constexpr uint32_t TILINGKEY_NO_QUANT = 0;
constexpr uint32_t TILINGKEY_STATIC_QUANT = 1;
constexpr uint32_t TILINGKEY_PERTOKEN_QUANT = 2;
constexpr uint32_t TILINGKEY_NO_FULLMESH = 0;
constexpr uint32_t TILINGKEY_ENABLE_FULLMESH = 1;
constexpr uint32_t TILINGKEY_ENABLE_HIERARCHY = 2;

template <bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh>
__global__ __aicore__ void moe_distribute_dispatch_v2_kernel(
    GM_ADDR mc2Context, GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales,
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut,
    GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut,
    GM_ADDR workspaceGM, MoeDistributeDispatchV2TilingData tilingData)
{
    TPipe pipe;

    if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
        if constexpr (QuantMode == TILINGKEY_NO_QUANT) {
            MoeDistributeDispatchV2FullMesh<bfloat16, bfloat16, UNQUANT, false, false> op;
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut,
                    dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM,
                    &pipe, &tilingData);
            op.Process();
        } else if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
            MoeDistributeDispatchV2FullMesh<bfloat16, int8_t, STATIC_QUANT, false, false> op;
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut,
                    dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM,
                    &pipe, &tilingData);
            op.Process();
        } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
            MoeDistributeDispatchV2FullMesh<bfloat16, int8_t, PERTOKEN_DYNAMIC_QUANT, ScaleMode, false> op;
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut,
                    dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM,
                    &pipe, &tilingData);
            op.Process();
        }
    } else if constexpr (FullMesh == TILINGKEY_NO_FULLMESH) {
        if constexpr (QuantMode == TILINGKEY_NO_QUANT) {
            MoeDistributeDispatchV2<bfloat16, bfloat16, UNQUANT, false, HasTp> op;
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut,
                    dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM,
                    &pipe, &tilingData);
            op.Process();
        } else if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
            MoeDistributeDispatchV2<bfloat16, int8_t, STATIC_QUANT, false, HasTp> op;
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut,
                    dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM,
                    &pipe, &tilingData);
            op.Process();
        } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
            MoeDistributeDispatchV2<bfloat16, int8_t, PERTOKEN_DYNAMIC_QUANT, ScaleMode, HasTp> op;
            op.Init(mc2Context, x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut,
                    dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM,
                    &pipe, &tilingData);
            op.Process();
        }
    }
}

void moe_distribute_dispatch_v2_entry(int32_t tilingKey, uint32_t blockDim, aclrtStream stream, void *mc2Context,
                                      void *x, void *expertIds, void *scales, void *xActiveMask, void *expertScales,
                                      void *elasticInfo, void *performanceInfo, void *expandXOut,
                                      void *dynamicScalesOut, void *assistInfoOut, void *expertTokenNumsOut,
                                      void *epSendCountsOut, void *tpSendCountsOut, void *expandScalesOut,
                                      void *workspace, const void *tilingData)
{
    const MoeDistributeDispatchV2TilingData *tiling =
        static_cast<const MoeDistributeDispatchV2TilingData *>(tilingData);

    uint8_t quantMode = tiling->moeDistributeDispatchV2Info.quantMode;
    uint8_t fullMesh = TILINGKEY_NO_FULLMESH;
    bool hasTp = false;
    bool scaleMode = false;

    moe_distribute_dispatch_v2_kernel<false, 0, false, 0><<<blockDim, nullptr, stream>>>(
        (GM_ADDR)mc2Context, (GM_ADDR)x, (GM_ADDR)expertIds, (GM_ADDR)scales, (GM_ADDR)xActiveMask,
        (GM_ADDR)expertScales, (GM_ADDR)elasticInfo, (GM_ADDR)performanceInfo, (GM_ADDR)expandXOut,
        (GM_ADDR)dynamicScalesOut, (GM_ADDR)assistInfoOut, (GM_ADDR)expertTokenNumsOut, (GM_ADDR)epSendCountsOut,
        (GM_ADDR)tpSendCountsOut, (GM_ADDR)expandScalesOut, (GM_ADDR)workspace, *tiling);
}