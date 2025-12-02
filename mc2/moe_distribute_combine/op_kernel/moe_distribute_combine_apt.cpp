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
 * \file moe_distribute_combine_apt.cpp
 * \brief
 */
#include "kernel_operator.h"
#include "lib/matmul_intf.h"
#include "arch35/moe_distribute_combine_arch35.h"
using namespace AscendC;

extern "C" __global__ __aicore__ void moe_distribute_combine(GM_ADDR expandX, GM_ADDR expertIds, GM_ADDR expandIdx,
                                                             GM_ADDR epSendCount, GM_ADDR scales, GM_ADDR tpSendCount,
                                                             GM_ADDR xActiveMask, GM_ADDR activationScale,
                                                             GM_ADDR weightScale, GM_ADDR groupList,
                                                             GM_ADDR expandScales, GM_ADDR XOut, GM_ADDR workspaceGM,
                                                             GM_ADDR tilingGM)

{
  GET_TILING_DATA_WITH_STRUCT(MoeDistributeCombineTilingDataA5, tilingData, tilingGM);


  TPipe pipe;
#if (ORIG_DTYPE_EXPAND_X == DT_BF16 || ORIG_DTYPE_EXPAND_X == DT_FLOAT16)
  if (TILING_KEY_IS(1000000000000000000)) {
    MoeDistributeCombineA5Impl::MoeDistributeCombineA5<DTYPE_EXPAND_X, int32_t> op;
    op.Init(expandX, expertIds, expandIdx, epSendCount, tpSendCount, nullptr, scales, nullptr, XOut, workspaceGM,
            &pipe, &tilingData);
    op.Process();
  }
#endif
}