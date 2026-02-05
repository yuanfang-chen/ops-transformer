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
#include "basic_api/kernel_basic_intf.h"
#include "log.h"

#if defined(__DAV_C310__)
#include "arch35/moe_distribute_dispatch_arch35.h"
#endif // defined(__DAV_C310__)

#if defined(__DAV_C310__)
using namespace MoeDistributeDispatchA5Impl;
#else
using namespace MoeDistributeDispatchA2Impl;
#endif

using namespace Mc2Tiling;
using namespace AscendC;
using namespace MoeDistributeDispatchV2HostKfcImpl;
using namespace MoeDispatchLog;

template<bool HasTp, uint8_t QuantMode, bool ScaleMode, uint8_t FullMesh, uint8_t CommMode, uint8_t ArchTag>
__global__ __aicore__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, 
    GM_ADDR elasticInfo, GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, 
    GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut, GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, 
    GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
//printf("kernel.cpp start");
LogInfo(__LINE__, "kernel_extend.cpp start");
REGISTER_TILING_DEFAULT(MoeDistributeDispatchV2TilingData);
#if defined(__DAV_C310__)
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchV2TilingData, tilingData, tilingGM);
#endif
    TPipe pipe;
    
#if defined(__DAV_C310__)
#if ((ORIG_DTYPE_EXPAND_X == DT_BF16) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT16))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (CommMode == TILINGKEY_TPL_HOST_KFC){
            LogInfo(__LINE__, "KFC start 16 UNQUANT");
            MoeDistributeDispatchV2HostKfc<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, false, false> op;
            op.Init(x,expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            LogInfo(__LINE__, "KFC end 16 UNQUANT");
        }
    } 
#elif ((ORIG_DTYPE_X == DT_FLOAT8_E5M2) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2)) ||   \
    ((ORIG_DTYPE_X == DT_FLOAT8_E4M3FN) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN)) || \
    ((ORIG_DTYPE_X == DT_HIFLOAT8) && (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (CommMode == TILINGKEY_TPL_HOST_KFC){
            LogInfo(__LINE__, "KFC start 8 UNQUANT");
            MoeDistributeDispatchV2HostKfc<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::UNQUANT, true, false> op;
            op.Init(x,expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                    expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut, workspaceGM, &pipe, &tilingData);
            op.Process();
            LogInfo(__LINE__, "KFC end 8 UNQUANT");
        }
    } 
#elif ((ORIG_DTYPE_EXPAND_X == DT_INT8) || (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2) || \
       (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN) || (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8))
    if constexpr (ArchTag == TILINGKEY_TPL_A5) {
        if constexpr (QuantMode == TILINGKEY_STATIC_QUANT) {
            if constexpr (CommMode == TILINGKEY_TPL_HOST_KFC){
                LogInfo(__LINE__, "KFC start 8 STATIC_QUANT");
                MoeDistributeDispatchV2HostKfc<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::STATIC_QUANT, true, false> op;
                op.Init(x,expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                LogInfo(__LINE__, "KFC end 8 STATIC_QUANT");
            }
        } else if constexpr (QuantMode == TILINGKEY_PERTOKEN_QUANT) {
            if constexpr (CommMode == TILINGKEY_TPL_HOST_KFC){
                LogInfo(__LINE__, "KFC start 8 PERTOKEN_DYNAMIC_QUANT");
                MoeDistributeDispatchV2HostKfc<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::PERTOKEN_DYNAMIC_QUANT, ScaleMode, false> op;
                op.Init(x,expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                LogInfo(__LINE__, "KFC end 8 PERTOKEN_DYNAMIC_QUANT");
            }
        } else if constexpr (QuantMode == TILINGKEY_PERGROUP_QUANT) {
            if constexpr (CommMode == TILINGKEY_TPL_HOST_KFC){
                LogInfo(__LINE__, "KFC start 8 PERGROUP_DYNAMIC_QUANT");
                MoeDistributeDispatchV2HostKfc<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::PERGROUP_DYNAMIC_QUANT, ScaleMode, false> op;
                op.Init(x,expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                LogInfo(__LINE__, "KFC end 8 PERGROUP_DYNAMIC_QUANT");
            }
        } else if constexpr (QuantMode == TILINGKEY_MX_QUANT) {
            if constexpr (CommMode == TILINGKEY_TPL_HOST_KFC){
                LogInfo(__LINE__, "KFC start 8 MX_QUANT");
                MoeDistributeDispatchV2HostKfc<DTYPE_X, DTYPE_EXPAND_X, MoeDistributeDispatchV2Impl::MX_QUANT, false, false> op;
                op.Init(x,expertIds, scales, xActiveMask, expertScales, elasticInfo, expandXOut, dynamicScalesOut, assistInfoOut,
                        expertTokenNumsOut, epSendCountsOut, tpSendCountsOut, expandScalesOut, workspaceGM, &pipe, &tilingData);
                op.Process();
                LogInfo(__LINE__, "KFC end 8 MX_QUANT");
            }
        } 
    } 
#endif
#endif
}