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
 * \brief
 */
#ifndef CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_DATA_H
#define CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_DATA_H

#include "kernel_tiling/kernel_tiling.h"

namespace ChunkGatedDeltaRuleRecurrence {
#pragma pack(push, 8)
struct alignas(8) ChunkGatedDeltaRuleRecurrenceTilingData {
    uint32_t coreNum;        // AIV core count
    uint32_t b;              // batch size
    uint32_t hv;             // value head count
    uint32_t realDk;         // actual dk
    uint32_t alignDk;        // aligned dk (multiple of FP32_NUM_PER_BLOCK=8)
    uint32_t realDv;         // actual dv
    uint32_t alignDv;        // aligned dv
    uint32_t nChunks;        // total chunk count across all batches
    uint32_t realChunkSize;  // actual chunk_size (cs)
    uint32_t alignChunkSize; // aligned chunk_size
    uint32_t dvTile;         // dv tile step (constrained by UB)
    uint32_t totalTasks;     // b * hv
    uint32_t tasksPerCore;   // ceil(totalTasks / coreNum)
    float scaleValue;        // scale attribute
};
#pragma pack(pop)
} // namespace ChunkGatedDeltaRuleRecurrence

#endif // CHUNK_GATED_DELTA_RULE_RECURRENCE_TILING_DATA_H
