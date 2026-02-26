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
 * \file moe_distribute_dispatch_v2_extend.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
// #include "moe_distribute_dispatch_v2.h"
// #include "moe_distribute_dispatch_v2_tiling.h"
// #include "moe_distribute_dispatch_v2_full_mesh.h"
// #include "moe_distribute_dispatch_v2_tiling_key.h"

#if __has_include("../common/inc/kernel/moe_distribute_base.h")
#include "../common/inc/mc2_moe_context.h"
#else 
#include "../../common/inc/mc2_moe_context.h"
#endif

// using namespace MoeDistributeDispatchV2Impl;
// using namespace MoeDistributeDispatchV2FullMeshImpl;
using namespace Mc2Context;
// using namespace Mc2Tiling;
using namespace AscendC;

template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch_v2_extend(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR mc2Context, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    __gm__ Mc2MoeContext * ptr = (__gm__ Mc2MoeContext *)(mc2Context);
    AscendC::printf("rankid %d",ptr->rankId);

// REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
// #if defined(__DAV_C310__)
//     GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
// #endif
//     TPipe pipe;

// #if defined(__DAV_C310__)
// #if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
//     Mc2MoeContext * ptr = (Mc2MoeContext *)(mc2context);
//     AscendC::printf("rankid %d",ptr->rankId);
//     if constexpr (ArchTag == TILINGKEY_TPL_A5) {
//         if constexpr (CommMode == TILINGKEY_TPL_CCU) {
//             MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
//             op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, 
//                     expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
//             op.Process();
//         } else if constexpr (CommMode == TILINGKEY_TPL_MTE) {
//             if constexpr (FullMesh == TILINGKEY_ENABLE_FULLMESH) {
//                 MoeDistributeDispatchV2FullMesh<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
//                 op.Init(x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
//                         expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
//                 op.Process();
//             } else {
//                 MoeDistributeDispatchV2<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
//                 op.Init(x, expertIds, scales, xActiveMask, elasticInfo, performanceInfo, expandXOut, dynamicScalesOut, assistInfoOut, 
//                         expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, workspaceGM, &pipe, &tilingData);
//                 op.Process();
//             }
//         }  
//     }
// #endif
// #endif
}