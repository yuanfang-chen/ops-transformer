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
 * \file moe_distribute_dispatch_setup.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "basic_api/kernel_basic_intf.h"
#include "arch35/moe_distribute_dispatch_teardown_arch35.h"
#include "moe_distribute_dispatch_teardown_tiling.h"
using namespace AscendC;
using namespace MoeDistributeDispatchTeardownImpl; 
extern "C" __global__ __aicore__ void moe_distribute_dispatch_teardown(
    GM_ADDR x, GM_ADDR y, GM_ADDR expertIds, GM_ADDR commCmdInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoForCombineOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchTeardownTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, UNQUANT, false, false> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, UNQUANT, false, true> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(10011)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, STATIC_QUANT, true, false> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, PERTOKEN_DYNAMIC_QUANT, false, false> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, PERTOKEN_DYNAMIC_QUANT, true, false> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11011)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, STATIC_QUANT, true, true> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, PERTOKEN_DYNAMIC_QUANT, false, true> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, PERTOKEN_DYNAMIC_QUANT, true, true> op;
         op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
            op.Process();
    }
#endif
}