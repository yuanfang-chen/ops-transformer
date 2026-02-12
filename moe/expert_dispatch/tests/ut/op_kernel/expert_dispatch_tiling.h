/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __EXPERT_DISPATCH_TILING_H__
#define __EXPERT_DISPATCH_TILING_H__

#include <cstdint>
#include <cstring>

#include "kernel_tiling/kernel_tiling.h"

#pragma pack(1)
struct ExpertVBSComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreElements = 0;
    int64_t lastCoreLoops = 0;
    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
    int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, ExpertVBSComputeTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(ExpertVBSComputeTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, ExpertVBSComputeTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ExpertVBSComputeTilingData));
}
#endif

#pragma pack(1)
struct ExpertVMSMiddleComputeTilingData {
    int64_t needCoreNum = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, ExpertVMSMiddleComputeTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(ExpertVMSMiddleComputeTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, ExpertVMSMiddleComputeTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ExpertVMSMiddleComputeTilingData));
}
#endif

#pragma pack(1)
struct ExpertSortOutComputeTilingData {
    int64_t oneLoopMaxElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void
InitTilingData(const __gm__ uint8_t* tiling, ExpertSortOutComputeTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(ExpertSortOutComputeTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void
InitTilingData(uint8_t* tiling, ExpertSortOutComputeTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ExpertSortOutComputeTilingData));
}
#endif

#pragma pack(1)
struct ExpertTokensCountTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreElements = 0;
    int64_t lastCoreElements = 0;
    int64_t perCoreLoops = 0;
    int64_t perCorePerLoopElements = 0;
    int64_t perCoreLastLoopElements = 0;
    int64_t lastCoreLoops = 0;

    int64_t lastCorePerLoopElements = 0;
    int64_t lastCoreLastLoopElements = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, ExpertTokensCountTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(ExpertTokensCountTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, ExpertTokensCountTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ExpertTokensCountTilingData));
}
#endif

#pragma pack(1)
struct ExpertGatherOutComputeTilingData {
    int64_t needCoreNum = 0;
    int64_t perCoreIndicesElements = 0;
    int64_t lastCoreIndicesElements = 0;
    int64_t perCoreIndicesLoops = 0;
    int64_t perCorePerLoopIndicesElements = 0;
    int64_t perCoreLastLoopIndicesElements = 0;
    int64_t lastCoreIndicesLoops = 0;
    int64_t lastCorePerLoopIndicesElements = 0;
    int64_t lastCoreLastLoopIndicesElements = 0;
    int64_t colsLoops = 0;
    int64_t perLoopCols = 0;
    int64_t lastLoopCols = 0;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, ExpertGatherOutComputeTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(ExpertGatherOutComputeTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, ExpertGatherOutComputeTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ExpertGatherOutComputeTilingData));
}
#endif

#pragma pack(1)
struct ExpertDispatchTilingData {
    int64_t coreNum = 0;
    int64_t n = 0;
    int64_t cols = 0;
    int64_t k = 0;
    int64_t expertStart = 0;
    int64_t expertEnd = 0;
    int64_t actualExpertNum = 0;
    ExpertVBSComputeTilingData vbsComputeParamsOp;
    ExpertVMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    ExpertSortOutComputeTilingData sortOutComputeParamsOp;
    ExpertTokensCountTilingData expertTokensCountTilingDataOp;
    ExpertGatherOutComputeTilingData gatherOutComputeParamsOp;
};
#pragma pack()

#ifdef __NPU_TILING__
inline[aicore] void InitTilingData(const __gm__ uint8_t* tiling, ExpertDispatchTilingData* const_data)
{
    const __gm__ uint32_t* src = (const __gm__ uint32_t*)tiling;
    uint32_t* dst = (uint32_t*)const_data;
    for (auto i = 0; i < sizeof(ExpertDispatchTilingData) / 4; i++) {
        *(dst + i) = *(src + i);
    }
}
#else
inline void InitTilingData(uint8_t* tiling, ExpertDispatchTilingData* const_data)
{
    memcpy(const_data, tiling, sizeof(ExpertDispatchTilingData));
}
#endif  // __NPU_TILING__

#define GET_TILING_DATA_WITH_STRUCT(tiling_struct, tiling_data, tiling_arg) \
    tiling_struct tiling_data;                                              \
    InitTilingData(tiling_arg, &tiling_data)
#define GET_TILING_DATA(tiling_data, tiling_arg) \
    ExpertDispatchTilingData tiling_data;        \
    InitTilingData(tiling_arg, &tiling_data)

#define DTYPE_X float
#endif  // __EXPERT_DISPATCH_TILING_H__
