/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\n * \file aggregate_hidden_grad_apt.cpp\n * \brief AICore entry for aggregate_hidden_grad (arch35)\n */
#include "kernel_operator.h"
#include "arch35/aggregate_hidden_grad.h"
#include "arch35/aggregate_hidden_grad_struct.h"

using namespace AscendC;
using AggregateHiddenGradKernelNS::AggregateHiddenGradKernel;
using AggregateHiddenGradArch35Tiling::AggregateHiddenGradTilingDataV35;

extern "C" __global__ __aicore__ void aggregate_hidden_grad(
    GM_ADDR grad_output, GM_ADDR input, GM_ADDR weight, GM_ADDR mask,
    GM_ADDR grad_input, GM_ADDR grad_weight,
    GM_ADDR workspace, GM_ADDR tiling)
{
    if (g_coreType == AIC) {
        return;
    }
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    REGISTER_TILING_DEFAULT(AggregateHiddenGradTilingDataV35);
    TPipe pipe;
    GET_TILING_DATA_WITH_STRUCT(AggregateHiddenGradTilingDataV35, td, tiling);

    if (TILING_KEY_IS(0)) {
        AggregateHiddenGradKernel<half> op;
        op.Init(grad_output, input, weight, mask, grad_input, grad_weight, &td, &pipe);
        op.Process();
    }
    
    else {
        AggregateHiddenGradKernel<bfloat16_t> op;
        op.Init(grad_output, input, weight, mask, grad_input, grad_weight, &td, &pipe);
        op.Process();
    }
   

    // Unsupported dtype key path

}
