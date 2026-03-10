/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __TEST_BLOCK_SPARSE_ATTENTION_TILING_H__
#define __TEST_BLOCK_SPARSE_ATTENTION_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#define __CCE_UT_TEST__
#define __aicore__

BEGIN_TILING_DATA_DEF(BlockSparseAttentionTilingData)
TILING_DATA_FIELD_DEF(uint32_t, batch);
TILING_DATA_FIELD_DEF(uint32_t, numHeads);
TILING_DATA_FIELD_DEF(uint32_t, kvHeads);
TILING_DATA_FIELD_DEF(uint32_t, embeddingSize);
TILING_DATA_FIELD_DEF(uint32_t, blockSize);
TILING_DATA_FIELD_DEF(uint32_t, maxNumBlocksPerBatch);
TILING_DATA_FIELD_DEF(uint32_t, firstBatchTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, totalTaskNum);
TILING_DATA_FIELD_DEF(uint32_t, maskType);
TILING_DATA_FIELD_DEF(float, scaleValue);
TILING_DATA_FIELD_DEF(uint32_t, totalQBlocks);
TILING_DATA_FIELD_DEF(uint32_t, firstQBlockNum);
TILING_DATA_FIELD_DEF(uint64_t, blockShapeX);
TILING_DATA_FIELD_DEF(uint64_t, blockShapeY);
TILING_DATA_FIELD_DEF(uint32_t, maxKvBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, maxQBlockNum);
TILING_DATA_FIELD_DEF(uint32_t, queryLayout);
TILING_DATA_FIELD_DEF(uint32_t, kvCacheLayout);
TILING_DATA_FIELD_DEF(uint32_t, maxQSeqlen);
TILING_DATA_FIELD_DEF(uint32_t, maxKvSeqlen);
TILING_DATA_FIELD_DEF(uint32_t, useUniformQSeqlen);
TILING_DATA_FIELD_DEF(uint32_t, useUniformKvSeqlen);
TILING_DATA_FIELD_DEF(uint64_t, tilingKey);
TILING_DATA_FIELD_DEF(uint64_t, selectNumIdxSize);
TILING_DATA_FIELD_DEF(uint64_t, selectIdxSize);
TILING_DATA_FIELD_DEF(uint64_t, mm1OutSize);
TILING_DATA_FIELD_DEF(uint64_t, smOnlineOutSize);
TILING_DATA_FIELD_DEF(uint64_t, mm2OutSize);
TILING_DATA_FIELD_DEF(uint64_t, updateSize);
TILING_DATA_FIELD_DEF(uint64_t, workSpaceSize);
END_TILING_DATA_DEF;

template <class T>
void InitBsaMemberData(uint8_t* tiling, T* const_data)
{
    memcpy(const_data, tiling, sizeof(T));
}

inline void InitBSATilingData(uint8_t* tiling, BlockSparseAttentionTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(BlockSparseAttentionTilingData));
}

#define GET_TILING_DATA(tilingData, tilingPointer) \
    BlockSparseAttentionTilingData tilingData;         \
    InitBSATilingData(tilingPointer, &tilingData)

#define GET_TILING_DATA_MEMBER(tiling_type, member, var, tiling) \
    REGISTER_TILINGDATA_SIZE(tiling_type, __COUNTER__); \
    decltype(((tiling_type *)0)->member) var; \
    size_t offset##var = (size_t)(&((tiling_type *)0)->member); \
    InitBsaMemberData<decltype(((tiling_type *)0)->member)>(tiling + offset##var, &var)

#endif
