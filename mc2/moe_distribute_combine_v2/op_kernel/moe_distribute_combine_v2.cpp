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
 * \file moe_distribute_combine_v2.cpp
 * \brief
 */
#include <vector>
#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "moe_distribute_combine_v2_tiling_key.h"

#ifdef __DAV_C310__
#include "arch35/moe_distribute_combine_arch35.h"
#else
#include "moe_distribute_combine_a2.h"
#include "moe_distribute_combine_a2_layered.h"
#include "moe_distribute_combine_a2_layered_aicpu.h"
#endif // __DAV_C310__
#include "moe_distribute_combine_v2_tiling.h"
#include "moe_distribute_combine_v2.h"

#ifndef __DAV_C310__
using namespace MoeDistributeCombineA2Impl;
#endif // __DAV_C310__

using namespace MoeDistributeCombineV2Impl;
using namespace Mc2Tiling;
using namespace AscendC;

namespace {
__aicore__ inline WinContext GetWinContext(const MoeDistributeDispatchV2TilingData *tilingData)
{
    WinContext winContext;

    epContext = (__gm__ Mc2Kernel::HcclOpParam*)AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
    winContext.commEp.localUsrRankId = epWinContext_->localUsrRankId;
    winContext.commEp.rankSize = epWinContext_->rankSize;
    winContext.commEp.getWinSize = epContext->winSize;
    winContext.commEp.getStatusDataSpaceGm = epContext->localWindowsExp;
    uint32_t epWorldSize = tilingData->moeDistributeDispatchV2Info.epWorldSize;
    winContext.commEp.windowInAddr.resieze(epWorldSize);
    winContext.commEp.windowExpAddr.resieze(epWorldSize);

    for (uint32_t rankId = 0; rankId < epWorldSize; ++rankId) {
        if (rankId == tilingData->moeDistributeDispatchV2Info.epRankId) {
            winContext.commEp.windowInAddr[rankId] = epContext->localWindowsIn;
            winContext.commEp.windowExpAddr[rankId] = epContext->localWindowsExp;
        } else {
            winContext.commEp.windowInAddr[rankId] = ((HcclRankRelationResV2 *)(epContext->remoteRes[rankId].nextDevicePtr))->windowsIn;
            winContext.commEp.windowExpAddr[rankId] = ((HcclRankRelationResV2 *)(epContext->remoteRes[rankId].nextDevicePtr))->windowsExp;
        }
    }

    tpContext = (__gm__ Mc2Kernel::HcclOpParam*)AscendC::GetHcclContext<1>();
    winContext.commTp.getWinSize = tpContext->winSize;
    uint32_t tpWorldSize = tilingData->moeDistributeDispatchV2Info.tpWorldSize;
    winContext.commTp.windowInAddr.resieze(tpWorldSize);
    winContext.commTp.windowExpAddr.resieze(tpWorldSize);

    for (uint32_t rankId = 0; rankId < tpWorldSize; ++rankId) {
        if (rankId == tilingData->moeDistributeDispatchV2Info.tpRankId) {
            winContext.commTp.windowInAddr[rankId] = tpContext->localWindowsIn;
            winContext.commTp.windowExpAddr[rankId] = tpContext->localWindowsExp;
        } else {
            winContext.commTp.windowInAddr[rankId] = ((HcclRankRelationResV2 *)(tpContext->remoteRes[rankId].nextDevicePtr))->windowsIn;
            winContext.commTp.windowExpAddr[rankId] = ((HcclRankRelationResV2 *)(tpContext->remoteRes[rankId].nextDevicePtr))->windowsExp;
        }
    }

    return winContext;
}

template <CombineMC2TypeClass>
__aicore__ inline void ExecMoeDistributeCombineV2(WinContext winContext, GM_ADDR expandX, GM_ADDR expertIds,
                                                GM_ADDR assistInfoForCombine, GM_ADDR epSendCount, 
                                                GM_ADDR tpSendCount, GM_ADDR scales, GM_ADDR xActiveMask, 
                                                GM_ADDR sharedExpertX, GM_ADDR elasticInfo, GM_ADDR oriX, 
                                                GM_ADDR constExpertAlpha1, GM_ADDR constExpertAlpha2,
                                                GM_ADDR constExpertV, GM_ADDR performanceInfo, GM_ADDR XOut, 
                                                GM_ADDR workspaceGM, GM_ADDR tilingGM, TPipe *pipePtr)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
    MoeDistributeCombineV2<CombineMC2TypeFunc> op;
    op.Init(winContext, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, nullptr, nullptr,
            scales, xActiveMask, sharedExpertX, elasticInfo, oriX, constExpertAlpha1, 
            constExpertAlpha2, constExpertV, performanceInfo, nullptr, nullptr, XOut, workspaceGM, pipePtr, &tilingData);
    op.Process();
}
}
template<bool HasTp, uint8_t QuantMode, uint8_t LayeredMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_combine_v2(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR assistInfoForCombine,
                                                    GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                    GM_ADDR xActiveMask, GM_ADDR activationScale, GM_ADDR weightScale,
                                                    GM_ADDR groupList, GM_ADDR expandScales, GM_ADDR sharedExpertX, 
                                                    GM_ADDR elasticInfo, GM_ADDR oriX, GM_ADDR constExpertAlpha1,
                                                    GM_ADDR constExpertAlpha2, GM_ADDR constExpertV, 
                                                    GM_ADDR performanceInfo, GM_ADDR XOut, GM_ADDR workspaceGM, 
                                                    GM_ADDR tilingGM)

{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineV2TilingData);
#ifndef __DAV_C310__
    REGISTER_TILING_FOR_TILINGKEY("ArchTag == TILINGKEY_TPL_A2", MoeDistributeCombineA2TilingData);
#endif
    TPipe pipe;

#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
#ifdef __DAV_C310__
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
        MoeDistributeCombineA5Impl::MoeDistributeCombineA5<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, xActiveMask,
                scales, sharedExpertX, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#else
    if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                (LayeredMode == TILINGKEY_TPL_MTE) && (QuantMode == TILINGKEY_NO_QUANT)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
        MoeDistributeCombineA2<DTYPE_EXPAND_X, int32_t> op;
        op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, scales, xActiveMask, oriX, constExpertAlpha1, 
                constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                        (QuantMode == TILINGKEY_NO_QUANT) && (LayeredMode == TILINGKEY_TPL_AICPU)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);
        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AICPU) {
            MoeDistributeCombineA2LayeredAicpu<DTYPE_EXPAND_X, int32_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, expandScales, 
                    XOut, workspaceGM, &pipe, &tilingData, contextGM0);
            op.Process();
        } else if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t, DTYPE_EXPAND_X> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, expandScales, performanceInfo, 
                    XOut, workspaceGM, &pipe, &tilingData, contextGM0);
            op.Process();
        }
    } else if constexpr ((ArchTag == TILINGKEY_TPL_A2) && 
                        (QuantMode == TILINGKEY_INT8_QUANT) && (LayeredMode == TILINGKEY_TPL_AICPU)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineA2TilingData, tilingData, tilingGM);

        auto contextGM0 = AscendC::GetHcclContext<HCCL_GROUP_ID_0>();
        DataplaneMode dataplaneMode = GetDataplaneMode(contextGM0);
        if (dataplaneMode == DataplaneMode::AIV) {
            MoeDistributeCombineA2Layered<DTYPE_EXPAND_X, int32_t, int8_t> op;
            op.Init(expandX, expertIds, assistInfoForCombine, epSendCount, expandScales, performanceInfo,
                    XOut, workspaceGM, &pipe, &tilingData, contextGM0);
            op.Process();
        }
    }
#endif
    if constexpr (ArchTag == TILINGKEY_TPL_A3) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineV2TilingData, tilingData, tilingGM);
        WinContext winContext = GetWinContext(&tilingData);
        ExecMoeDistributeCombineV2<DTYPE_EXPAND_X, DTYPE_X, int32_t, HasTp, QuantMode == TILINGKEY_INT8_QUANT, false>(
        winContext, expandX, expertIds, assistInfoForCombine, epSendCount, tpSendCount, scales, xActiveMask, sharedExpertX, 
        elasticInfo, oriX, constExpertAlpha1, constExpertAlpha2, constExpertV, performanceInfo, XOut, workspaceGM, tilingGM, &pipe);
    }
#endif
}