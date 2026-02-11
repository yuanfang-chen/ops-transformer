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
 * \file mhc_post.h
 * \brief MhcPost kernel implementation
 * Formula: y = (H_res)^T * x + h_out * h_post
 */

#ifndef ASCENDC_MHC_POST_H
#define ASCENDC_MHC_POST_H

#include "kernel_tiling/kernel_tiling.h"
#include "kernel_operator.h"
#include "kernel_operator_intf.h"

namespace MhcPost {
using namespace AscendC;

constexpr uint32_t BLOCK_NUM_FP16 = 16;    // 16 elements per block for FP16 (32 bytes)
constexpr uint32_t BLOCK_NUM_FP32 = 8;     // 8 elements per block for FP32 (32 bytes)
constexpr uint32_t BLOCK_NUM_BF16 = 16;    // 16 elements per block for BF16 (32 bytes)

/**
 * @brief Get aligned length in elements for block alignment
 */
template <typename T>
__aicore__ inline uint32_t GetAlignedLength(uint32_t dataLen)
{
    uint32_t blockNum = 0;
    if (std::is_same<T, float>::value) {
        blockNum = BLOCK_NUM_FP32;
    } else if (std::is_same<T, half>::value || std::is_same<T, bfloat16_t>::value) {
        blockNum = BLOCK_NUM_FP16;
    }
    return ((dataLen + blockNum - 1) / blockNum) * blockNum;
}

} // namespace MhcPost

#endif // ASCENDC_MHC_POST_H