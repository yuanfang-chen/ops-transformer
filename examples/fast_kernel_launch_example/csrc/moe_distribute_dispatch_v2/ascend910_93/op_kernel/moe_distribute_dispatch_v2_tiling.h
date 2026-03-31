/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You can not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file moe_distribute_dispatch_v2_tiling.h
 * \brief Simplified TilingData for direct launch mode (A3 only)
 */

#ifndef MOE_DISTRIBUTE_DISPATCH_V2_DIRECT_TILING_H
#define MOE_DISTRIBUTE_DISPATCH_V2_DIRECT_TILING_H

#include <cstdint>

struct MoeDistributeDispatchV2Info {
    uint32_t epWorldSize;
    uint32_t tpWorldSize;
    uint32_t epRankId;
    uint32_t tpRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t quantMode;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t a;
    uint32_t aivNum;
    bool isTokenMask;
    bool isExpertMask;
    bool hasElasticInfo;
    bool isPerformance;
    bool isQuant;
    bool isMc2Context;
    bool reserved1;
    bool reserved2;
    uint64_t totalUbSize;
    uint64_t totalWinSizeEp;
    uint64_t totalWinSizeTp;
    uint32_t expertTokenNumsType;
    int32_t zeroComputeExpertNum;
    uint32_t maxSizeForUbBuffer;
    uint64_t scalesRow;
    uint64_t scalesCol;
    uint32_t scalesTypeSize;
    uint64_t scalesCount;
};

struct MoeDistributeDispatchV2TilingData {
    MoeDistributeDispatchV2Info moeDistributeDispatchV2Info;
};

#endif