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
        uint32_t vectorCoreNum;
        float scale;
        uint32_t hasGamma;
    };
    #pragma pack(pop)
}  // ChunkGatedDeltaRule

#endif  // CHUNK_GATED_DELTA_RULE_TILING_DATA_H