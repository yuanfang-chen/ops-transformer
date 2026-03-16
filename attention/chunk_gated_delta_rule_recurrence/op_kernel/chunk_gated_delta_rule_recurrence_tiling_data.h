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
 * \file chunk_gated_delta_rule_recurrence_tiling_data.h
 * \brief TilingData for ChunkGatedDeltaRuleRecurrence (fused Cube+Vector, ascend950 only)
 *
 * Matmul shapes:
 *   C12 (C1=vPrime, C2=attn):  A=[cs,dk]  B=[dv,dk](transposeB)  C=[cs,dv]
 *   C3  (delta = vNew^T@kgexp): A=[cs,dv](transposeA) B=[cs,dk]  C=[dv,dk]
 */
#ifndef CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace ChunkGatedDeltaRuleRecurrence {
#pragma pack(push, 8)
struct alignas(8) ChunkGatedDeltaRuleRecurrenceTilingData {
    uint32_t coreNum;        // AIV core count
    uint32_t coreNumAic;     // AIC core count
    uint32_t b;              // batch size
    uint32_t hv;             // value head count
    uint32_t realDk;         // actual dk
    uint32_t alignDk;        // aligned dk (multiple of FP32_PER_BLOCK=8)
    uint32_t realDv;         // actual dv
    uint32_t alignDv;        // aligned dv
    uint32_t nChunks;        // total chunk count across all batches
    uint32_t realChunkSize;  // actual chunk_size (cs)
    uint32_t alignChunkSize; // aligned chunk_size
    uint32_t dvTile;         // dv tile step for AIV (constrained by UB)
    uint32_t totalTasks;     // b * hv
    uint32_t tasksPerCore;   // ceil(totalTasks / coreNumAic)
    uint32_t wsPerGroup;     // workspace floats per AIC group
    float    scaleValue;     // scale attribute

    // Cube tiling for C1/C2: kCumdecay/qgexp [cs,dk] × state[dv,dk]^T → [cs,dv]
    TCubeTiling cubeTilingC12;
    // Cube tiling for C3: vNew[cs,dv]^T × kgexp[cs,dk] → [dv,dk]
    TCubeTiling cubeTilingC3;
};
#pragma pack(pop)
} // namespace ChunkGatedDeltaRuleRecurrence

#endif // CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_DATA_H
