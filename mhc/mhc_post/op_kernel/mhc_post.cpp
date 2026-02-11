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
 * Formula: y = (H_res)^T * x + h_out * h_post
 */

#include "mhc_post.h"

#define TILING_KEY_MHC_POST_FP16 1
#define TILING_KEY_MHC_POST_BF16 2

extern "C" __global__ __aicore__ void mhc_post(GM_ADDR x, GM_ADDR h_res, GM_ADDR h_out,
                                                GM_ADDR h_post, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    (void)workspace;

    GET_TILING_DATA_WITH_STRUCT(MhcPostTilingData, tilingDataIn, tiling);
    const MhcPostTilingData* __restrict__ tilingData = &tilingDataIn;

    if (TILING_KEY_IS(TILING_KEY_MHC_POST_FP16)) {
        MhcPost::MhcPostKernel<half, float> op;
        op.Init(x, h_res, h_out, h_post, y, tilingData);
        op.Process();
    } else if (TILING_KEY_IS(TILING_KEY_MHC_POST_BF16)) {
        MhcPost::MhcPostKernel<bfloat16_t, float> op;
        op.Init(x, h_res, h_out, h_post, y, tilingData);
        op.Process();
    }
}