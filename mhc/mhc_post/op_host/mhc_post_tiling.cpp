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
 * \file mhc_post_tiling.cpp
 * \brief MhcPost tiling implementation
 */

#include "mhc_post_tiling.h"
#include "register/op_tiling_registry.h"

namespace optiling {
/**
 * Get the tiling data for MhcPost operator
 */
void MhcPostTilingFunc(const ge::Operator &op, ge::TensorDesc *inputDescs,
                       ge::TensorDesc *outputDescs, const std::vector<int64_t> &workspaces,
                       std::vector<int64_t> &blockDims, std::vector<void *> &workspaceList) {
    (void)workspaces;
    (void)workspaceList;

    // Get input tensor shape
    ge::Shape xShape = inputDescs[0].GetShape();
    uint32_t totalLength = 1;
    for (size_t i = 0; i < xShape.GetDimNum(); ++i) {
        totalLength *= xShape.GetDim(i);
    }

    // Calculate core num and block dims
    uint32_t coreNum = 1;
    uint32_t singleCoreLength = totalLength;

    constexpr uint32_t MIN_LENGTH_PER_CORE = 32;
    constexpr uint32_t MAX_CORE_NUM = 40; // Ascend910B has 40 AIV cores

    if (totalLength >= MIN_LENGTH_PER_CORE * MAX_CORE_NUM) {
        coreNum = MAX_CORE_NUM;
        singleCoreLength = (totalLength + coreNum - 1) / coreNum;
        // Align to 32 elements for better performance
        singleCoreLength = (singleCoreLength + 31) / 32 * 32;
    } else if (totalLength >= MIN_LENGTH_PER_CORE) {
        coreNum = totalLength / MIN_LENGTH_PER_CORE;
        if (coreNum < 1) {
            coreNum = 1;
        }
        if (coreNum > MAX_CORE_NUM) {
            coreNum = MAX_CORE_NUM;
        }
        singleCoreLength = (totalLength + coreNum - 1) / coreNum;
        singleCoreLength = (singleCoreLength + 31) / 32 * 32;
    }

    // Set block dims for multi-core parallel
    blockDims.push_back(coreNum);
}
} // namespace optiling

REGISTER_OP_TILING_V3(MhcPost, optiling::MhcPostTilingFunc)