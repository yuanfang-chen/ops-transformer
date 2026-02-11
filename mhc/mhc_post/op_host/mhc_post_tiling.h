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
 * \file mhc_post_tiling.h
 * \brief MhcPost tiling header
 */

#ifndef MHC_POST_TILING_H
#define MHC_POST_TILING_H

#include "register/op_def_registry.h"
#include "register/tilingdata_base.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"
#include "log/log.h"

namespace optiling {

constexpr int64_t FLOAT_DATA_SIZE = 4;
constexpr int64_t BLOCK_NUM_FP16 = 16;
constexpr int64_t BLOCK_NUM_BF16 = 16;
constexpr int64_t BLOCK_NUM_FP32 = 8;
constexpr int64_t MIN_BUFFER_NUM = 2;
constexpr int64_t ALIGN_256 = 256;

// Tiling keys for different data types
constexpr int64_t TILING_KEY_FP16 = 1;
constexpr int64_t TILING_KEY_BF16 = 2;

struct MhcPostTilingParams {
    int64_t totalLength = 0;      // Total elements in x
    int64_t coreNum = 0;          // Number of cores
    int64_t singleCoreLength = 0; // Elements per core
    int64_t tileLength = 0;       // Tile length for each iteration
    int64_t maxCoreMemery = 0;   // Max UB size
};

ge::graphStatus TilingComputeForMhcPost(gert::TilingContext* context, MhcPostTilingParams& param);

BEGIN_TILING_DATA_DEF(MhcPostTilingData)
TILING_DATA_FIELD_DEF(int64_t, total_length);
TILING_DATA_FIELD_DEF(int64_t, core_num);
TILING_DATA_FIELD_DEF(int64_t, single_core_length);
END_TILING_DATA_DEF

REGISTER_TILING_DATA_CLASS(MhcPost, MhcPostTilingData)

struct MhcPostCompileInfo {
};

} // namespace optiling
#endif // MHC_POST_TILING_H