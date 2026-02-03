/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "block_sparse_attention_grad_tiling.h"
#include <cmath>
#include <cstring>
#include "log/log.h"

#include <cstdint>
#include <string>
#include "err/ops_err.h"
#include "graph/types.h"
#include "graph/tensor.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling_base/tiling_base.h"

using namespace ge;
using namespace std;

constexpr int TND_DIM_T = 0;
constexpr int TND_DIM_N = 1;
constexpr int TND_DIM_D = 2;
constexpr int TND_DIM_NUM = 3;

constexpr int BNSD_DIM_B = 0;
constexpr int BNSD_DIM_N = 1;
constexpr int BNSD_DIM_S = 2;
constexpr int BNSD_DIM_D = 3;
constexpr int BNSD_DIM_NUM = 4;

constexpr int BSH_DIM_B = 0;
constexpr int BSH_DIM_S = 1;
constexpr int BSH_DIM_H = 2;

constexpr int QUERY_INDEX = 0;
constexpr int KEY_INDEX = 1;
constexpr int VALUE_INDEX = 2;
constexpr int SELECT_IDX_INDEX = 3;
constexpr int SELECT_NUM_IDX_INDEX = 4;
constexpr int BLOCK_SHAPE_INDEX = 5;

constexpr int ATTENTION_MASK_INDEX = 6;
constexpr int ACTUAL_SEQ_LENGTHS_INDEX = 7;
constexpr int ACTUAL_SEQ_LENGTHS_KV_INDEX = 8;
constexpr int BLOCK_TABLE_INDEX = 9;
constexpr int MAX_BLOCK_NUM_INDEX = 2;


constexpr int Q_INPUT_LAYOUT_INDEX = 0;
constexpr int KV_INPUT_LAYOUT_INDEX = 1;
constexpr int NUM_KEY_VALUE_HEADS_INDEX = 2;
constexpr int MASK_TYPE_INDEX = 3;
constexpr int SCALE_VALUE_INDEX = 4;
constexpr int INNER_PRECISE_INDEX = 5;
constexpr int BLOCK_SIZE_INDEX = 6;

constexpr int VALID_EMBEDDING_SIZE_64 = 64;
constexpr int VALID_EMBEDDING_SIZE_128 = 128;

namespace optiling {

constexpr uint32_t BASIC_BLOCK_SIZE = 128;
constexpr uint32_t WORKSPACE_BLOCK_SIZE_DB = 131072;
constexpr uint32_t NUM3 = 3;

static inline uint32_t CeilDiv(uint32_t n1, uint32_t n2)
{
    if (n1 == 0) {
        return 0;
    }
    return (n2 != 0) ? ((n1 + n2 - 1) / n2) : n1;
}

static inline uint32_t GetQBlocks(int32_t qseqlen, int32_t x)
{
    uint32_t qBlocksInX = CeilDiv(x, BASIC_BLOCK_SIZE);
    uint32_t completeXBlocks = x != 0 ? qseqlen / x : qseqlen / BASIC_BLOCK_SIZE;
    uint32_t remainingSeqlen = x != 0 ? qseqlen - completeXBlocks * x : qseqlen % BASIC_BLOCK_SIZE;
    uint32_t remainingBlocks = CeilDiv(remainingSeqlen, BASIC_BLOCK_SIZE);
    return qBlocksInX * completeXBlocks + remainingBlocks;
}

ASCENDC_EXTERN_C ge::graphStatus TilingBlockSparseAttentionGrad(gert::TilingContext* context)
{
    OP_CHECK_IF(context == nullptr, OPS_REPORT_VECTOR_INNER_ERR("RainFusionAttention",
        "Context is nullptr."), return ge::GRAPH_FAILED);

    BlockSparseAttentionGradTilingData tilingData;
    return ge::GRAPH_SUCCESS;
}

ASCENDC_EXTERN_C ge::graphStatus TilingPrepareBlockSparseAttentionGrad(gert::TilingParseContext* context)
{
    (void) context;
    return ge::GRAPH_SUCCESS;
}

IMPL_OP_OPTILING(BlockSparseAttentionGrad)
    .Tiling(TilingBlockSparseAttentionGrad)
    .TilingParse<BlockSparseAttentionGradCompileInfo>(TilingPrepareBlockSparseAttentionGrad); // 向框架注册入口函数;

}  // namespace optiling
