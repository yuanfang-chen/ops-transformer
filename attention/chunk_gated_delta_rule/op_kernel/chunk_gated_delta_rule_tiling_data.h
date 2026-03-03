/**
* This program is free software, you can redistribute it and/or modify.
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file grouped_matmul_finalize_routing.h
 * \brief
 */

#ifndef __CHUNK_GATED_DELTA_RULE_TILING_DATA_H__
#define __CHUNK_GATED_DELTA_RULE_TILING_DATA_H__

#include "kernel_tiling/kernel_tiling.h"

namespace ChunkGatedDeltaRule {
    #pragma pack(push, 8)
    struct alignas(8) ChunkGatedDeltaRuleTilingData { 
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
    };
    #pragma pack(pop)

    struct ChunkGroup {
        int64_t startPos = 0;    // 该ChunkGroup在T上的起始位置
        int64_t length = 0;      // 该ChunkGroup的长度
        int64_t chunkSize = 0;   // 每个chunk的长度
        int64_t coreStart = 0;   // 预留
        int64_t coreEnd = 0;     // 预留
    };
}  // ChunkGatedDeltaRule

#endif  // CHUNK_GATED_DELTA_RULE_TILING_DATA_H