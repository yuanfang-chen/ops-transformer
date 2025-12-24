/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file allto_all_matmul.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch32/allto_all_matmul_tiling.h"
#include "arch32/allto_all_matmul.h"

using namespace AscendC;
using namespace AlltoAllMatmulImpl;

extern "C" __global__ __aicore__ void allto_all_matmul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR comm_scale,
                                            GM_ADDR x1_offset, GM_ADDR x2_offset, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    if (TILING_KEY_IS(1000000)) {
        KERNEL_TASK_TYPE(1000000, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);
        AlltoAllMatmul<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT, false> op;
        op.Init(x1, x2, bias, nullptr, nullptr, y, all2all_out, workspaceGM, tilingGM);
        op.Process();
    } else if (TILING_KEY_IS(1100000)) {
        KERNEL_TASK_TYPE(1100000, KERNEL_TYPE_MIX_AIC_1_2);
        GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);
        AlltoAllMatmul<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X1_SCALE, DTYPE_X2_SCALE, DTYPE_Y, DTYPE_ALL2ALL_OUT, true> op;
        op.Init(x1, x2, bias, nullptr, nullptr, y, all2all_out, workspaceGM, tilingGM);
        op.Process();
    }
}