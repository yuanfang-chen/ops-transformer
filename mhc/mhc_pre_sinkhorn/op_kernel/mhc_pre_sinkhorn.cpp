/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * the CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file mhc_pre_sinkhorn.cpp
 * \brief mhc_pre_sinkhorn
 */

#include "kernel_operator.h"
#include "op_kernel/math_util.h"
#include "op_kernel/platform_util.h"
#include "mhc_pre_sinkhorn_struct.h"
#include "mhc_pre_sinkhorn_tiling_key.h"
#include "arch35/mhc_pre_sinkhorn.h"

using namespace AscendC;
using namespace MhcPreSinkhorn;

ASCENDC_TPL_ARGS_DECL(MhcPreSinkhorn,
    ASCENDC_TPL_UINT_DECL(TILING_KEY, 1, ASCENDC_TPL_UI_LIST, 1)
);

ASCENDC_TPL_SEL(
    ASCENDC_TPL_ARGS_SEL(
        ASCENDC_TPL_KERNEL_TYPE_SEL(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_SEL(TILING_KEY, ASCENDC_TPL_UI_LIST, 1)
    )
);

template <uint32_t TILING_KEY>
__global__ __aicore__ void mhc_pre_sinkhorn(GM_ADDR h_res, GM_ADDR h_res_sinkhorn, GM_ADDR norm_out, GM_ADDR sum_out,
                                            GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    GET_TILING_DATA(tilingData, tilingGM);
    __gm__ uint8_t *user = GetUserWorkspace(workspaceGM);

    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    TPipe pipe;

    InitParams initParams{h_res, h_res_sinkhorn, norm_out, sum_out, user, &pipe, tilingData};
    MhcPreSinkhornSimd op(pipe, tilingData);
    op.Init(initParams);
    op.Process();
}

ASCENDC_TPL_ARGS_REG(
    ASCENDC_TPL_ARGS_REG(
        ASCENDC_TPL_KERNEL_TYPE_REG(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_REG(TILING_KEY, ASCENDC_TPL_UI_LIST, 1)
    )
);

ASCENDC_TPL_FUNC_REG(
    ASCENDC_TPL_ARGS_REG(
        ASCENDC_TPL_KERNEL_TYPE_REG(ASCENDC_TPL_AIV_ONLY),
        ASCENDC_TPL_UINT_REG(TILING_KEY, ASCENDC_TPL_UI_LIST, 1)
    ),
    ASCENDC_TPL_FUNC_NAME(mhc_pre_sinkhorn),
    ASCENDC_TPL_FUNC_PARAM(GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR, GM_ADDR)
);
