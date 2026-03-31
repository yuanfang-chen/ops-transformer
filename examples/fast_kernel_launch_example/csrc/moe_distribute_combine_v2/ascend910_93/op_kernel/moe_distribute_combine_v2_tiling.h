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
 * \file moe_distribute_combine_v2_tiling.h
 * \brief Simplified TilingData for direct launch mode (A3 only)
 */

#ifndef MOE_DISTRIBUTE_COMBINE_V2_DIRECT_TILING_H
#define MOE_DISTRIBUTE_COMBINE_V2_DIRECT_TILING_H

#include <cstdint>

struct MoeDistributeCombineV2Info {
    uint32_t epWorldSize;
    uint32_t tpWorldSize;
    uint32_t epRankId;
    uint32_t tpRankId;
    uint32_t expertShardType;
    uint32_t sharedExpertNum;
    uint32_t sharedExpertRankNum;
    uint32_t moeExpertNum;
    uint32_t moeExpertPerRankNum;
    uint32_t zeroExpertNum;
    uint32_t copyExpertNum;
    uint32_t constExpertNum;
    uint32_t globalBs;
    uint32_t bs;
    uint32_t k;
    uint32_t h;
    uint32_t a;
    uint32_t aivNum;
    bool isTokenMask;
    bool isExpertMask;
    bool hasSharedExpertX;
    bool hasElasticInfo;
    bool isPerformance;
    bool isMc2Context;
    bool reserved1;
    bool reserved2;
    uint64_t totalUbSize;
    uint64_t totalWinSizeEp;
    uint64_t totalWinSizeTp;
    float armAvgFactor;
    float epsilon;
    uint32_t bufferNum;
};

struct MoeDistributeCombineV2TilingData {
    MoeDistributeCombineV2Info moeDistributeCombineV2Info;
};

#endif