/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/* !
 * \file all_gather_matmul.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
#include "lib/matmul_intf.h"
#include "all_gather_matmul_tiling.h"
#include "all_gather_matmul_tiling_key.h"
#include "all_gather_matmul_inner.h"

using namespace AscendC;
using aType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, A_DTYPE, true>; 
using cType = MatmulType<AscendC::TPosition::GM, CubeFormat::ND, C_DTYPE>; 

__global__ __aicore__ void all_gather_matmul_inner(GM_ADDR aGM, GM_ADDR bGM, GM_ADDR contextGM, GM_ADDR biasGM,
    GM_ADDR cGM, GM_ADDR gatherOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
    REGISTER_TILING_DEFAULT(AllGatherMatmulInnerTilingData);

    GET_TILING_DATA_WITH_STRUCT(AllGatherMatmulInnerTilingData, tilingData, tilingGM);
    AllGatherMatmulInner<DTYPE_X1, DTYPE_X2, DTYPE_BIAS, DTYPE_Y, __VA_ARGS__> op;
    op.Init(aGM, bGM, contextGM, biasGM, cGM, gatherOut, workspaceGM, tilingGM);
    op.Process();
}