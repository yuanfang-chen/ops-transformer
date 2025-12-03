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
 * \file matmul_reduce_scatter_v2.cpp
 * \brief
 */

#include "lib/matmul_intf.h"
#include "kernel_operator.h"
#include "matmul_reduce_scatter_aiv_mode.h"
#include "matmul_reduce_scatter_v2_aiv_mode_tiling.h"
#include "matmul_reduce_scatter_v2_aiv_tiling_key.h"

using namespace matmulReduceScatterV2_aivmode_tiling;
using namespace MatmulReduceScatterv2TilingKey;
using namespace AscendC;
using namespace MatmulReduceScatterV2Impl;

template<bool IsBias, bool IsTransposeA, bool IsTransposeB>
__global__ __aicore__ void matmul_reduce_scatter_v2(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR biasGM,
                                                               GM_ADDR x1ScaleGM, GM_ADDR x2ScaleGM,
                                                               GM_ADDR quantScaleGM, GM_ADDR cGM, GM_ADDR amaxOutGM,
                                                               GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
//aiv算子模板
    #define INVOKE_MMREDUCESCATTER_AIV_MODE_OP_IMPL(templateClass, ...)                                              \
        do {                                                                                                    \
            GET_TILING_DATA_WITH_STRUCT(MatmulReduceScatterV2AivModeTilingData, tilingData, tilingGM);          \
            templateClass<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_X2_SCALE, DTYPE_Y, __VA_ARGS__> op;                             \
            op.Init(aGM, bGM, biasGM, x1ScaleGM, x2ScaleGM, cGM, workspaceGM, tilingGM);                        \
            op.Process();                                                                                       \
        } while (0)

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(MatmulReduceScatterV2AivModeTilingData);
    if constexpr (!IsBias && !IsTransposeA && !IsTransposeB) {
        //aivMode，非transB
        INVOKE_MMREDUCESCATTER_AIV_MODE_OP_IMPL(MatmulReduceScatterAivMode, FORMAT_X2 == FORMAT_FRACTAL_NZ, false, false);
    } else if constexpr (!IsBias && !IsTransposeA && IsTransposeB) {
        //aivMode，transB
        INVOKE_MMREDUCESCATTER_AIV_MODE_OP_IMPL(MatmulReduceScatterAivMode, FORMAT_X2 == FORMAT_FRACTAL_NZ, false, true);
    }
}