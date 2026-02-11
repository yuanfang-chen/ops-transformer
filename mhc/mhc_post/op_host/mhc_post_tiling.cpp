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
 * Formula: y = (H_res)^T * x + h_out * h_post
 * where: h_res is [M, K] (FP32), x is [batch, ..., M] (FP16/BF16)
 *       result of (H_res)^T * x is [K]
 *       h_out is same shape as x, h_post is scalar
 */
void MhcPostTilingFunc(const ge::Operator &op, ge::TensorDesc *inputDescs,
                       ge::TensorDesc *outputDescs, const std::vector<int64_t> &workspaces,
                       std::vector<int64_t> &blockDims, std::vector<void *> &workspaceList) {
    (void)workspaces;
    (void)workspaceList;
    (void)outputDescs;

    // Get input tensor shapes
    ge::Shape xShape = inputDescs[0].GetShape();  // x: [batch, ..., M]
    ge::Shape hResShape = inputDescs[1].GetShape(); // h_res: [M, K] or scalar
    ge::Shape hOutShape = inputDescs[2].GetShape(); // h_out: same as x
    ge::Shape hPostShape = inputDescs[3].GetShape(); // h_post: scalar or same as x

    uint32_t xDimNum = xShape.GetDimNum();
    uint32_t hResDimNum = hResShape.GetDimNum();

    // M is the last dimension of x
    uint32_t m = xShape.GetDim(xDimNum - 1);
    uint32_t k = 1;  // Default to scalar

    // K is the last dimension of h_res if it's 2D
    if (hResDimNum >= 2) {
        k = hResShape.GetDim(hResDimNum - 1);
    }

    // Calculate total elements in x for element-wise multiplication
    uint32_t totalLength = 1;
    for (size_t i = 0; i < xDimNum; ++i) {
        totalLength *= xShape.GetDim(i);
    }

    // Calculate core num and block dims for element-wise operations
    uint32_t coreNum = 1;
    uint32_t singleCoreLength = totalLength;

    constexpr uint32_t MIN_LENGTH_PER_CORE = 256;
    constexpr uint32_t MAX_CORE_NUM = 32; // Ascend950 has 32 AIV cores

    if (totalLength >= MIN_LENGTH_PER_CORE * MAX_CORE_NUM) {
        coreNum = MAX_CORE_NUM;
        singleCoreLength = (totalLength + coreNum - 1) / coreNum;
        // Align to 256 elements for better performance
        singleCoreLength = (singleCoreLength + 255) / 256 * 256;
    } else if (totalLength >= MIN_LENGTH_PER_CORE) {
        coreNum = totalLength / MIN_LENGTH_PER_CORE;
        if (coreNum < 1) {
            coreNum = 1;
        }
        if (coreNum > MAX_CORE_NUM) {
            coreNum = MAX_CORE_NUM;
        }
        singleCoreLength = (totalLength + coreNum - 1) / coreNum;
        singleCoreLength = (singleCoreLength + 255) / 256 * 256;
    }

    // Set block dims for multi-core parallel
    blockDims.push_back(coreNum);
}

REGISTER_OP_TILING_V3(MhcPost, optiling::MhcPostTilingFunc)
} // namespace optiling