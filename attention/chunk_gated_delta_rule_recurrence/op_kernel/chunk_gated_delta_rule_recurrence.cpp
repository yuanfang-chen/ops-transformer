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
 * \file chunk_gated_delta_rule_recurrence.cpp
 * \brief Kernel entry point for ChunkGatedDeltaRuleRecurrence
 */
#include "chunk_gated_delta_rule_recurrence.h"

using namespace AscendC;
using namespace ChunkGatedDeltaRuleRecurrence;

extern "C" __global__ __aicore__ void chunk_gated_delta_rule_recurrence(
    __gm__ uint8_t *initialState,
    __gm__ uint8_t *kgexp,
    __gm__ uint8_t *value,
    __gm__ uint8_t *kCumdecay,
    __gm__ uint8_t *qgexp,
    __gm__ uint8_t *gexp,
    __gm__ uint8_t *cuSeqlens,
    __gm__ uint8_t *attnInterOut,
    __gm__ uint8_t *vNewOut,
    __gm__ uint8_t *workspace,
    __gm__ uint8_t *tiling)
{
    REGISTER_TILING_DEFAULT(ChunkGatedDeltaRuleRecurrenceTilingData);
    GET_TILING_DATA(tilingData, tiling);
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    TPipe pipe;
    CGDR op(&tilingData);
    op.Init(initialState, kgexp, value, kCumdecay, qgexp, gexp,
            cuSeqlens, attnInterOut, vNewOut, &pipe);
    op.Process();
}
