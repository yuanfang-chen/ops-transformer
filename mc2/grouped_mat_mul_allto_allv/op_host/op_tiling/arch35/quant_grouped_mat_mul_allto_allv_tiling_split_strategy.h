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
 * \file quant_grouped_mat_mul_allto_allv_tiling_split_strategy.h
 * \brief
 */

#ifndef QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_SPLIT_STRATEGY_H
#define QUANT_GROUPED_MAT_MUL_ALLTO_ALLV_TILING_SPLIT_STRATEGY_H

#pragma once
#include "quant_grouped_mat_mul_allto_allv_tiling.h"


namespace optiling {
namespace Mc2GroupedMatmul {

// enum class QuantMode : uint8_t {
//     NON_QUANT = 0, // 非量化模式
//     TT_QUANT = 1,  // TT量化模式
//     ERROR = 255    // 特殊设置，表示不支持的类型组合
// };

class QuantGroupedMatmulAllToAllvTilingStrategyTT {
public:
    // static QuantMode GetQuantMode(const gert::TilingContext *context);
    // static ge::graphStatus GetSplitNums(const gert::TilingContext *context);
    uint32_t mSizePerLoop = 0;
};

} // namespace MC2Tiling
}
#endif
