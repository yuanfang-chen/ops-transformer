/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2024-2025. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*!
 * \file moe_distribute_dispatch_teardown.cpp
 * \brief
 */

#include "kernel_operator.h"
#include "moe_distribute_dispatch_teardown.h"
#include "../moe_distribute_dispatch_teardown/moe_distribute_dispatch_teardown_tiling.h"

using namespace AscendC;
using namespace MoeDistributeDispatchTeardownImpl;

/*
 * A3 tilingkey说明
 * 5位的十进制数
 * 第1位（个位）：quantMode:
 *     0: 不量化, 1: 静态量化, 2: 动态量化
 * 第2位（十位）：是否有smoothScale:
 *     0: 无, 1: 有
 * 第3位（百位）：是否做tp域allgather:
 *     0: 不做, 1: 做
 * 第4位（千位）：是否是共享专家卡:
 *     0: 不是, 1: 是
 * 第5位（万位）：无实际意义
 */

extern "C" __global__ __aicore__ void moe_distribute_dispatch_teardown(
    GM_ADDR x, GM_ADDR y, GM_ADDR expertIds, GM_ADDR commCmdInfo, GM_ADDR expandXOut, GM_ADDR dynamicScalesOut,
    GM_ADDR assistInfoForCombineOut, GM_ADDR expertTokenNumsOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeDispatchTeardownTilingData);
    TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
    if (TILING_KEY_IS(10000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, false, false, false, false> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11000)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, false, false, false, true> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#elif (ORIG_DTYPE_EXPAND_X == DT_INT8)
    if (TILING_KEY_IS(10011)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, true, false, false, false> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, false, true, false, false> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, false, true, true, false> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11011)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, true, false, false, true> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11002)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, false, true, false, true> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11012)) {
        GET_TILING_DATA_WITH_STRUCT(MoeDistributeDispatchTeardownTilingData, tilingData, tilingGM);
        MoeDistributeDispatchTeardown<DTYPE_X, DTYPE_EXPAND_X, false, true, true, true> op;
        op.Init(
            x, y, expertIds, commCmdInfo, expandXOut, dynamicScalesOut, assistInfoForCombineOut, expertTokenNumsOut,
            workspaceGM, &pipe, &tilingData);
        op.Process();
    }
#endif
}
