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
 * \file mhc_post_apt.cpp
 * \brief MhcPost kernel entry point
 * Formula: x_{l+1} = (H_{l}^{res})^{T} * x_l + h_{l}^{out} * H_{t}^{post}
 */

#include "kernel_operator.h"
#include "kernel_tiling/kernel_tiling.h"
#include "arch35/mhc_post.h"

using namespace AscendC;
using namespace MhcPost;

template <uint16_t isNAligned, uint16_t isNNAligned, uint16_t isDAligned>
__global__ __aicore__ void mhc_post(GM_ADDR x, GM_ADDR hRes, GM_ADDR hOut, GM_ADDR hPost, GM_ADDR output,
                                    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }

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

    REGISTER_TILING_DEFAULT(MhcPostTilingData);
    GET_TILING_DATA_WITH_STRUCT(MhcPostTilingData, tilingData, tiling);
    TPipe tPipe;

    MhcPostKernel<DTYPE_X, isNAligned, isNNAligned, isDAligned> op;
    op.Init(x, hRes, hOut, hPost, output, workspace, &tilingData, &tPipe);
    op.Process();

    tPipe.Destroy();
}