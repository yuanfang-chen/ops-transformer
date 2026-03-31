/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!
 * \file all_gather_add_tiling_data.h
 * \brief TilingData 结构体定义
 */

#ifndef ALL_GATHER_ADD_TILING_DATA_H
#define ALL_GATHER_ADD_TILING_DATA_H

#include <kernel_tiling/kernel_tiling.h>

struct AllGatherAddTilingInfo {
    int64_t rankCount;                 // all-gather 参与的 rank 数
    int64_t turnCount;                 // 按轮执行的总轮次数
    int64_t inputElementsPerRank;      // 单个 rank 上输入 A 的总元素数
    int64_t elementsPerTurnPerRank;    // 非尾轮时，每个 rank 每轮处理的元素数
    int64_t lastTurnElementsPerRank;   // 最后一轮每个 rank 实际处理的元素数
    int64_t outputElements;            // 输出 a_gathered / c 的总元素数
    int64_t elementsPerCorePerTurn;    // 单轮内每个核最多负责的元素数
    int64_t tileElementsPerCoreCalc;   // 单核单次计算 tile 的元素数
};

struct AllGatherAddTilingData {
    Mc2InitTiling mc2InitTiling;
    Mc2CcTiling mc2CcTiling;
    AllGatherAddTilingInfo allGatherAddTilingInfo;
};

#endif // ALL_GATHER_ADD_TILING_DATA_H
