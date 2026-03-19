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
 * \file moe_distribute_dispatch_tiling.h
 * \brief
 */

#ifndef ASCENDC_MOE_DISTRIBUTE_DISPATCH_TILING_H
#define ASCENDC_MOE_DISTRIBUTE_DISPATCH_TILING_H

#include <cstdint>
#include "kernel_tiling/kernel_tiling.h"

struct MoeDistributeDispatchTilingData {
    uint32_t epWorldSize = 2;                // epWorldSize
    uint32_t tpWorldSize = 1;                // tpWorldSize
    uint32_t epRankId = 0;                   // epRankId
    uint32_t tpRankId = 0;                   // tpRankId
    uint32_t expertShardType = 0;            // expert type
    uint32_t sharedExpertRankNum = 0;        // shared expert number
    uint32_t moeExpertNum = 2;               // moe expert number
    uint32_t quantMode = 0;                  // quant mode
    uint32_t globalBs = 16;                   // globalBs = BS * worldSize
    uint32_t bs = 8;                         // bs
    uint32_t k = 2;                          // k
    uint32_t h = 7168;                          // h
    uint32_t a = 16;                          // a
    uint32_t aivNum = 56;                     // aivNum
    bool isQuant = 0;                        // whether quant or not
    bool reserved1 = false;                      // reserved
    bool reserved2 = false;                      // reserved
    bool reserved3 = false;                      // reserved
    uint64_t totalUbSize = 253952;                // epWorldSize
    uint64_t totalWinSizeEp = 5242880;
    uint64_t totalWinSizeTp = 0;
    uint32_t expertTokenNumsType = 0;        // expert token nums type, support 0: cumsum mode, 1: count mode
};

#endif