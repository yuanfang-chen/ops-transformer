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
 * \file mhc_post.cpp
 * \brief MhcPost kernel entry point
 * Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "mhc_post.h"

using namespace AscendC;
using namespace MhcPost;

extern "C" __global__ __aicore__ void mhc_post(
    GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost,
    GM_ADDR output, GM_ADDR workspace, GM_ADDR tiling)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);

    if (g_coreType == AIC) {
        return;
    }

    (void)workspace;

    GET_TILING_DATA(tilingData, tiling);

    TPipe tPipe;

    // Template instantiation based on alignment flags from tiling data
    // IS_N_ALIGNED: n % 8 == 0 (n=4,8: true; n=6: false)
    // IS_NN_ALIGNED: (n*n) % 8 == 0 (16,64: true; 36: false)
    // IS_D_ALIGNED: D % 16 == 0 && nTilesD == 1
    //
    // Note: IS_N_ALIGNED and IS_NN_ALIGNED are always同步变化:
    // - n=4: isNAligned=0, isNNAligned=1
    // - n=6: isNAligned=0, isNNAligned=0
    // - n=8: isNAligned=1, isNNAligned=1
    //
    // Actually for n=4: 4%8!=0 but 16%8==0, so isNAligned=0, isNNAligned=1
    // So we need to handle all 8 combinations properly

    uint32_t isNAligned = tilingData.isNAligned;
    uint32_t isNNAligned = tilingData.isNNAligned;
    uint32_t isDAligned = tilingData.isDAligned;

    // Use 3-bit encoding: (isNAligned << 2) | (isNNAligned << 1) | isDAligned
    uint32_t templateKey = (isNAligned << 2) | (isNNAligned << 1) | isDAligned;

    switch (templateKey) {
        case 0: {  // 000: N不对齐, NN不对齐, D不对齐 (最慢路径, e.g., n=6, D=2561)
            MhcPostKernel<DTYPE_X, 0, 0, 0> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
        case 1: {  // 001: N不对齐, NN不对齐, D对齐 (e.g., n=6, D=2560)
            MhcPostKernel<DTYPE_X, 0, 0, 1> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
        case 2: {  // 010: N不对齐, NN对齐, D不对齐 (e.g., n=4, D=2561)
            MhcPostKernel<DTYPE_X, 0, 1, 0> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
        case 3: {  // 011: N不对齐, NN对齐, D对齐 (e.g., n=4, D=2560)
            MhcPostKernel<DTYPE_X, 0, 1, 1> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
        case 4: {  // 100: N对齐, NN不对齐, D不对齐 (理论上不可能: n=8时NN必对齐)
            MhcPostKernel<DTYPE_X, 1, 0, 0> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
        case 5: {  // 101: N对齐, NN不对齐, D对齐 (理论上不可能)
            MhcPostKernel<DTYPE_X, 1, 0, 1> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
        case 6: {  // 110: N对齐, NN对齐, D不对齐 (e.g., n=8, D=2561)
            MhcPostKernel<DTYPE_X, 1, 1, 0> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
        case 7: {  // 111: N对齐, NN对齐, D对齐 (最快路径, e.g., n=8, D=2560)
            MhcPostKernel<DTYPE_X, 1, 1, 1> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
        default: {
            // Fallback to slowest path
            MhcPostKernel<DTYPE_X, 0, 0, 0> op;
            op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
            op.Process();
            break;
        }
    }

    tPipe.Destroy();
}