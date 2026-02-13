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
 * \file qbmm_reduce_scatter_add_rms_norm_cast.cpp
 * \brief
 */

#include "basic_api/kernel_basic_intf.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_data.h"
// #include "qbmm_reduce_scatter_add_rms_norm_cast_tiling_key.h"
#include "qbmm_reduce_scatter_add_rms_norm_cast_mte.h"

using namespace AscendC;
using namespace QbmmReduceScatterAddRmsNormCastImpl;

__global__ __aicore__ void qbmm_reduce_scatter_add_rms_norm_cast(GM_ADDR x1, GM_ADDR x2, GM_ADDR y, GM_ADDR gamma, 
                                                                 GM_ADDR scale, GM_ADDR bias,  GM_ADDR perTokenScale,
                                                                 GM_ADDR y1, GM_ADDR y2, GM_ADDR x,
                                                                 GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    PRINTF("Goes into kernel 30. %d\n", 23);
    REGISTER_TILING_DEFAULT(QbmmReduceScatterAddRmsNormCastTilingData);
    GET_TILING_DATA_WITH_STRUCT(QbmmReduceScatterAddRmsNormCastTilingData, tilingData, tilingGM);
    TPipe pipe;
    if (TILING_KEY_IS(0)) {
        const QbmmReduceScatterAddRmsNormCastTilingData *qBmmReduceScatterAddRmsNormCastTilingData = &tilingData;                               \
        const TCubeTiling *mmTiling = &(qBmmReduceScatterAddRmsNormCastTilingData->matmulTiling);
        QbmmReduceScatterAddRmsNormCastMte <DTYPE_X1, DTYPE_Y, DTYPE_SCALE, false> op;
        REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), op.mm, mmTiling);
        op.Init(x, x2, y, gamma, scale,  bias, perTokenScale, y1, 
                y2, x, &pipe, workspaceGM, &tilingData);
        op.Process();
    }
}
