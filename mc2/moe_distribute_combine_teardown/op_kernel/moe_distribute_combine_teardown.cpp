/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, 
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_combine_teardown.cpp
 * \brief
 */
#include "basic_api/kernel_basic_intf.h"
#include "moe_distribute_combine_teardown_tiling.h"

using namespace AscendC;

extern "C" __global__ __aicore__ void
moe_distribute_combine_teardown(GM_ADDR expandX, GM_ADDR quantExpandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                GM_ADDR expertScales, GM_ADDR commCmdInfo, GM_ADDR xActiveMask,
                                GM_ADDR sharedExpertX, GM_ADDR XOut, GM_ADDR workspaceGM, GM_ADDR tilingGM)
{
    REGISTER_TILING_DEFAULT(MoeDistributeCombineTeardownTilingData); 
    GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTeardownTilingData, tilingData, tilingGM);
}