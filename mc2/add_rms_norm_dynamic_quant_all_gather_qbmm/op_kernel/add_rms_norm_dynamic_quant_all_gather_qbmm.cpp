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
 * \file add_rms_norm_dynamic_quant_v2.cpp
 * \brief
 */

#include "lib/matmul_intf.h"
#include "basic_api/kernel_basic_intf.h"
#include "add_rms_norm_dynamic_quant_all_gather_qbmm.h"
#include "add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h"

static constexpr uint32_t HARD_SYNC = 0;
static constexpr uint32_t NO_TILE_K = 1;
static constexpr uint32_t SOFT_SYNC = 2;

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant_all_gather_qbmm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, GM_ADDR smooth_scale,
    GM_ADDR bias, GM_ADDR output, GM_ADDR z, GM_ADDR dynamicQuantOut, GM_ADDR allGatherDataOut,
    GM_ADDR allGatherScalesOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(AddRmsNormDynamicQuantAllGatherQbmmTilingData);
    GET_TILING_DATA_MEMBER(AddRmsNormDynamicQuantAllGatherQbmmTilingData, tilingInfo, tilingData, tilingGM);
    if (TILING_KEY_IS(10000)) { // 切K硬同步+中间过程结果不输出+no smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, HARD_SYNC, false, false> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10001)) { // 切K硬同步+中间过程结果不输出+smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, HARD_SYNC, false, true> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10010)) { // 切K硬同步+中间过程结果全输出+no smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, HARD_SYNC, true, false> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10011)) { // 切K硬同步+中间过程结果全输出+smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, HARD_SYNC, true, true> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10100)) { // 不切K串行+中间过程结果不输出+no smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, NO_TILE_K, false, false> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10101)) { // 不切K串行+中间过程结果不输出+smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, NO_TILE_K, false, true> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10110)) { // 不切K串行+中间过程结果全输出+no smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, NO_TILE_K, true, false> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(10111)) { // 不切K串行+中间过程结果全输出+smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, NO_TILE_K, true, true> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11000)) { // 切K软同步+中间过程结果不输出+no smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, SOFT_SYNC, false, false> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11001)) { // 切K软同步+中间过程结果不输出+smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, SOFT_SYNC, false, true> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11010)) { // 切K软同步+中间过程结果全输出+no smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, SOFT_SYNC, true, false> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    } else if (TILING_KEY_IS(11011)) { // 切K软同步+中间过程结果全输出+smoothScale
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, DTYPE_SCALE, SOFT_SYNC, true, true> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, dynamicQuantOut,
                allGatherDataOut, allGatherScalesOut, workspaceGM, &pipe, &tilingData);
        op.Process();
    }
}