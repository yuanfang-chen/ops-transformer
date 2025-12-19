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
 * \file matmul_allto_all.cpp
 * \brief kernel内核实现
 */
#include <kernel_operator.h>
#include <lib/matmul_intf.h>
#if __CCE_AICORE__ == 220
#include "arch32/matmul_allto_all_tiling.h"
#include "arch32/matmul_allto_all.h"
#else
#include "arch35/matmul_allto_all_tiling_data.h"
#include "arch35/matmul_allto_all_tiling_key.h"
#include "arch35/matmul_allto_all.h"
#endif //__CCE_AICORE__ == 220

using namespace AscendC;
using namespace MatmulAlltoAllImpl;

extern "C" __global__ __aicore__ void matmul_allto_all(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR comm_scale,
                                                       GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MatmulAlltoAllTilingData);
#if __CCE_AICORE__ == 220
    if (TILING_KEY_IS(1000000)) {
        KERNEL_TASK_TYPE(1000000, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);
        MatmulAlltoAll<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, false, false> op;
        op.Init(x1, x2, bias, y, workspaceGM, tilingGM);
        op.Process();
    }

    if (TILING_KEY_IS(1000001)) {
        KERNEL_TASK_TYPE(1000001, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);
        MatmulAlltoAll<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, true, false> op;
        op.Init(x1, x2, bias, y, workspaceGM, tilingGM);
        op.Process();
    }

    if (TILING_KEY_IS(1000010)) {
        KERNEL_TASK_TYPE(1000010, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);
        MatmulAlltoAll<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, false, true> op;
        op.Init(x1, x2, bias, y, workspaceGM, tilingGM);
        op.Process();
    }

    if (TILING_KEY_IS(1000011)) {
        KERNEL_TASK_TYPE(1000011, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);
        MatmulAlltoAll<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, true, true> op;
        op.Init(x1, x2, bias, y, workspaceGM, tilingGM);
        op.Process();
    }
#else
    GET_TILING_DATA_WITH_STRUCT(MatmulAlltoAllTilingData, tilingData, tilingGM);
#endif //__CCE_AICORE__ == 220
}