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
 * \file chunk_gated_delta_rule_tiling_data.h
 * \brief
 */

#ifndef __CHUNK_GATED_DELTA_RULE_TILING_DATA_H__
#define __CHUNK_GATED_DELTA_RULE_TILING_DATA_H__

#include "kernel_tiling/kernel_tiling.h"

namespace ChunkGatedDeltaRule {
    constexpr uint64_t STRUCT_ALIGNAS = 8;
    #pragma pack(push, 8)
    struct alignas(STRUCT_ALIGNAS) ChunkGatedDeltaRuleTilingData {
        int64_t aiCoreNum;
        int64_t t;
        int64_t nk;
        int64_t dk;
        int64_t nv;
        int64_t dv;
        int64_t b;
        int64_t hasGamma;
        int64_t chunkSize;
        int64_t maxGroupLength;    // maxGroupLength = p * chunkSize
        int64_t interWorkspaceSz;
        int64_t stageWorkspaceSz;
        float scale;
        AscendC::tiling::TCubeTiling matmulTilingFp32;  // for MT_FP32: fp32 -> fp32
    };
    #pragma pack(pop)

    struct ChunkGroup {
        int64_t startPos = 0;    // 该ChunkGroup在T上的起始位置
        int64_t length = 0;      // 该ChunkGroup的长度
        int64_t chunkSize = 0;   // 每个chunk的长度
        int64_t coreStart = 0;   // 预留
        int64_t coreEnd = 0;     // 预留
    };

    // 同步信号
    constexpr uint64_t V_MTE3_EVENT = 0;
    constexpr uint64_t V_S_EVENT = 1;
    constexpr uint64_t MTE2_V_EVENT = 2;
    constexpr uint64_t S_V_EVENT = 3;
    constexpr uint64_t MTE3_MTE2_EVENT = 4;
    constexpr uint64_t FIX_MTE2_EVENT = 6;

    constexpr uint64_t BUFFER_NUM_ONE = 1;
    constexpr uint64_t BROADCAST_AXIS = 2;
    constexpr uint64_t TASK_RATIO = 2;
    constexpr uint64_t STAGE3_BUFFER_COUNT = 4;
    constexpr uint32_t MAX_L0_SIZE = 64 * 1024; // 64KB
    constexpr uint32_t BLOCK_SIZE = 32;         // copypad对齐块大小
    constexpr uint32_t BLOCK_FLOAT_NUM = 8;
    constexpr uint32_t BLOCK_BF16_NUM = 16;
}  // ChunkGatedDeltaRule

#endif  // CHUNK_GATED_DELTA_RULE_TILING_DATA_H