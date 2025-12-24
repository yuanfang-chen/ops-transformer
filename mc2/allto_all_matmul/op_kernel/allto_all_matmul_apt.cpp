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
 * \file allto_all_matmul.cpp
 * \brief kernel内核实现
 */
#include <kernel_operator.h>
#include <lib/matmul_intf.h>
#include "arch35/allto_all_matmul_tiling_data.h"
#include "arch35/allto_all_matmul_tiling_key.h"
#include "arch35/allto_all_matmul.h"

using namespace AscendC;
using namespace AlltoAllMatmulImpl;

template <uint32_t QUANT_MODE, bool X2_TRANSPOSE, uint32_t BIAS_DTYPE>
__global__ __aicore__ void allto_all_matmul(GM_ADDR x1, GM_ADDR x2, GM_ADDR bias, GM_ADDR x1_scale, GM_ADDR x2_scale, GM_ADDR comm_scale, GM_ADDR x1_offset,
                                            GM_ADDR x2_offset, GM_ADDR y, GM_ADDR all2all_out, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(AlltoAllMatmulTilingData);
    GET_TILING_DATA_WITH_STRUCT(AlltoAllMatmulTilingData, tilingData, tilingGM);
}