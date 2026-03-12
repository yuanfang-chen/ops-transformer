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
 * \file mhc_pre.cpp
 * \brief
 */

#include "arch35/mhc_pre.h"

using namespace AscendC;
using namespace matmul;
using namespace MhcPre;

extern "C" __global__ __aicore__ void mhc_pre(GM_ADDR x, GM_ADDR phi, GM_ADDR alpha, GM_ADDR bias, GM_ADDR gamma,
                                              GM_ADDR hin, GM_ADDR h_post, GM_ADDR h_res, GM_ADDR inv_rms,
                                              GM_ADDR h_mix, GM_ADDR h_pre, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;
    InitParams initParams{x,     phi,     alpha, bias,  gamma, hin,   h_post,
                          h_res, inv_rms, h_mix, h_pre, user,  &pipe, &tilingData};
    if (TILING_KEY_IS(0UL)) {
        MT mm;
        mm.Init(&tilingData.matmulTiling, &pipe);
        MhcPreKernel<DTYPE_X, float32_t> op(mm);
        op.Init(initParams);
        op.Process();
    }
}
