/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "../moe_distribute_dispatch/arch35/moe_distribute_dispatch_arch35.h"

using namespace MoeDistributeDispatchA5Impl;
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_distribute_dispatch_v2(
    GM_ADDR x, GM_ADDR expertIds, GM_ADDR scales, GM_ADDR xActiveMask, GM_ADDR expertScales, GM_ADDR elasticInfo,
    GM_ADDR performanceInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut, GM_ADDR assistInfoOut, GM_ADDR expertTokenNumsOut,
    GM_ADDR epSendCountsOut, GM_ADDR tpSendCountsOut, GM_ADDR expandScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTilingDataA5, tilingData, tilingGM);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(1000000000000000000)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, UNQUANT_MODE, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#elif ((ORIG_DTYPE_X == DT_FLOAT8_E5M2) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2)) ||   \
    ((ORIG_DTYPE_X == DT_FLOAT8_E4M3FN) && (ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN)) || \
    ((ORIG_DTYPE_X == DT_HIFLOAT8) && (ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8))
    if (TILING_KEY_IS(1000000000000000010)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, UNQUANT_MODE, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8 || ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E5M2 || \
       ORIG_DTYPE_EXPAND_X == DT_FLOAT8_E4M3FN || ORIG_DTYPE_EXPAND_X == DT_HIFLOAT8)
    if (TILING_KEY_IS(1000000000000000011)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, STATIC_QUANT_MODE, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000002)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, DYNAMIC_QUANT_MODE, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000012)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, DYNAMIC_QUANT_MODE, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000003)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, MXFP8_E5M2_QUANT_MODE, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000004)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, MXFP8_E4M3_QUANT_MODE, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000005)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, FP8_E5M2_PERTOKEN_QUANT_MODE, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000015)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, FP8_E5M2_PERTOKEN_QUANT_MODE, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000006)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, FP8_E4M3_PERTOKEN_QUANT_MODE, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000016)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, FP8_E4M3_PERTOKEN_QUANT_MODE, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000007)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, FP8_E5M2_PERTILE_QUANT_MODE, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000017)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, FP8_E5M2_PERTILE_QUANT_MODE, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000008)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, FP8_E4M3_PERTILE_QUANT_MODE, false, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000018)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, FP8_E4M3_PERTILE_QUANT_MODE, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(1000000000000000019)) {
        MoeDistributeDispatchA5<DTYPE_X, DTYPE_EXPAND_X, HIF8_PERTENSOR_QUANT_MODE, true, false> op;
        op.Init(x, expertIds, scales, xActiveMask, expandXOut, dynamicScalesOut, assistInfoOut, expertTokenNumsOut, epSendCountsOut,
                tpSendCountsOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}