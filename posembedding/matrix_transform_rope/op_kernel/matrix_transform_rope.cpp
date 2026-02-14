/**
 * This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to License for details. You may not use this file except in compliance with License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
 * BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE. See LICENSE in the root of
 * the software repository for the full text of the License.
 */

/*!
 * \file matrix_transform_rope.cpp
 * \brief MatrixTransformRope 算子 kernel 实现
 */

#include "matrix_transform_rope.h"

using namespace AscendC;
using namespace matmul;
using namespace MatrixTransformRope;

extern "C" {
__global__ void matrix_transform_rope(GM_ADDR x, GM_ADDR cos, GM_ADDR sin, GM_ADDR rotate, GM_ADDR out,
                                      GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tiling);
    GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);

    if (TILING_KEY_IS(3011)) {
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using bT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using MT = matmul::MatmulImpl<aT, bT, cT>;
        MT mm;

        TPipe pipe;
        KERNEL_TASK_TYPE(3011, KERNEL_TYPE_MIX_AIC_1_2);
        RotateMatrixAll<float, float, MT> op(mm);
        op.Init(x, cos, sin, rotate, y, usrWorkspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3012)) {
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
        using bT = MatmulType<TPosition::GM, CubeFormat::ND, half>;
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using MT = matmul::MatmulImpl<aT, bT, cT>;
        MT mm;

        TPipe pipe;
        KERNEL_TASK_TYPE(3012, KERNEL_TYPE_MIX_AIC_1_2);
        RotateMatrixAll<half, half, MT> op(mm);
        op.Init(x, cos, sin, rotate, y, usrWorkspace, tilingData, &pipe);
        op.Process();
    } else if (TILING_KEY_IS(3013)) {
        using aT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
        using bT = MatmulType<TPosition::GM, CubeFormat::ND, bfloat16_t>;
        using cT = MatmulType<TPosition::GM, CubeFormat::ND, float>;
        using MT = matmul::MatmulImpl<aT, bT, cT>;
        MT mm;

        TPipe pipe;
        KERNEL_TASK_TYPE(3013, KERNEL_TYPE_MIX_AIC_1_2);
        RotateMatrixAll<bfloat16_t, bfloat16_t, MT> op(mm);
        op.Init(x, cos, sin, rotate, y, usrWorkspace, tilingData, &pipe);
        op.Process();
    }
}
}
