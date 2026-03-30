/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_TILING_H_
#define _FUSED_K_RMS_NORM_ROPE_STORE_KV_CACHE_MX_QUANT_TILING_H_

#include <cstring>
#include "kernel_tiling/kernel_tiling.h"

#define DT_BF16 bfloat16_t
#define ORIG_DTYPE_START DT_BF16
#define __CCE_UT_TEST__

#pragma pack(1)

#ifndef DTYPE_QKV
#define DTYPE_QKV bfloat16_t
#endif
#ifndef DTYPE_Q
#define DTYPE_Q int8_t
#endif

struct FusedKRmsNormRopeStoreKvCacheMxQuantTilingData {
    int64_t seqLengthSum;
    int64_t qkvNumHead;
    int64_t qNumHead;
    int64_t kNumHead;
    int64_t vNumHead;
    int64_t headDim;
    int64_t blockNum;
    int64_t blockSize;
    int64_t qUsedCoreNum;
    int64_t qBlockFactor;
    int64_t qUbFactor;
    int64_t kUsedCoreNum;
    int64_t kBlockFactor;
    int64_t kUbFactor;
    int64_t vUsedCoreNum;
    int64_t vBlockFactor;
    int64_t vTUbFactor;
    int64_t vNumHeadUbFactor;
    float epsilon;
    float reciprocal;
};

#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t *tiling, FusedKRmsNormRopeStoreKvCacheMxQuantTilingData *const_data)
{
    const __gm__ uint32_t *src = (const __gm__ uint32_t *)tiling;
    uint32_t *dst = (uint32_t *)const_data;
    for (auto i = 0U; i < sizeof(FusedKRmsNormRopeStoreKvCacheMxQuantTilingData) / sizeof(uint32_t); ++i) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t *tiling, FusedKRmsNormRopeStoreKvCacheMxQuantTilingData *const_data)
{
    std::memcpy(const_data, tiling, sizeof(FusedKRmsNormRopeStoreKvCacheMxQuantTilingData));
}
#endif

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)

#endif
