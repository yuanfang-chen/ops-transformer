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
// #include "add_rms_norm_dynamic_quant_v2_normal_kernel.h"
#include "add_rms_norm_dynamic_quant_all_gather_qbmm_tiling_data.h"

// #define INVOKE_ADD_RMS_NORM_DYNAMIC_QUANT_QUANT_ALL_GATHER_OP_IMPL(templateClass, isTransB, ...)                  \
//     do {                                                                                                          \
//         using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, false>;                         \
//         using bType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, B_DTYPE, isTransB>;                      \
//         using biasType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, typename BiasType<BIAS_DTYPE>::type>; \
//         using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>;                                \
//         REGISTER_TILING_DEFAULT(Mc2Tiling::AddRmsNormDynamicQuantAllGatherQbmmTilingData);                        \
//         auto tiling = (__gm__ Mc2Tiling::AddRmsNormDynamicQuantAllGatherQbmmTilingData*)tilingGM;                 \
//         GET_TILING_DATA(tilingData, tilingGM);                                                                    \
//         templateClass<aType, bType, biasType, cType> op;                                                          \
//         op.Init(x1, x2, residual, y, gamma, scale, smoothScale, bias, output, z, &tilingData, &pipe);             \
//         op.Process();                                                                                             \
//     } while (0)

extern "C" __global__ __aicore__ void add_rms_norm_dynamic_quant_all_gather_qbmm(
    GM_ADDR x1, GM_ADDR x2, GM_ADDR residual, GM_ADDR y, GM_ADDR gamma, GM_ADDR scale, GM_ADDR smooth_scale,
    GM_ADDR bias, GM_ADDR output, GM_ADDR z, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    TPipe pipe;
    REGISTER_TILING_DEFAULT(AddRmsNormDynamicQuantAllGatherQbmmTilingData);
    GET_TILING_DATA_WITH_STRUCT(AddRmsNormDynamicQuantAllGatherQbmmTilingData, tilingData, tilingGM);
    // GET_TILING_DATA(tilingData, tiling);
    // GM_ADDR usrWorkspace = AscendC::GetUserWorkspace(workspace);
    if (TILING_KEY_IS(1)) {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_1);
        AddRmsNormDynamicQuantAllGatherQbmmImpl::AddRmsNormDynamicQuantAllGatherQbmm<DTYPE_X1, false, false> op;
        op.Init(x1, x2, residual, y, gamma, scale, smooth_scale, bias, output, z, workspaceGM, &pipe, &tilingData);
        op.Process();
        AscendC::PRINTF("kernel TILING_KEY_IS(1) !!!");
    }
    // if (TILING_KEY_IS(0)) {
    //     // 0 Tiling, Do Nothing.
    //     INVOKE_ADD_RMS_NORM_DYNAMIC_QUANT_QUANT_ALL_GATHER_OP_IMPL(AddRmsNormDynamicQuantAllGatherQbmm, false);
    // } else(TILING_KEY_IS(1)) {
    //     INVOKE_ADD_RMS_NORM_DYNAMIC_QUANT_QUANT_ALL_GATHER_OP_IMPL(AddRmsNormDynamicQuantAllGatherQbmm, true);
    // }
    // kernel: addrmsnormdynamicquant -> allgather -> quantbmm

}