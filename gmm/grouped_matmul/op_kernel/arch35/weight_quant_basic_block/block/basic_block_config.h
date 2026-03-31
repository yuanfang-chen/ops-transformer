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
 * \file basic_block_config.h
 * \brief
 */
#ifndef GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_CONFIG_H
#define GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_CONFIG_H

#if ASC_DEVKIT_MAJOR >= 9
#include "kernel_basic_intf.h"
#else
#include "kernel_operator.h"
#include "kernel_operator_intf.h"
#endif
#include "lib/matmul_intf.h"
#include "../prologue/tool.h"

namespace WeightQuantBatchMatmulV2::Arch35 {

struct WqmmConfig {
    bool aTrans;
    bool bTrans;
    CubeFormat weightFormat;
};

// kernel/block共用
struct BasicBlockOffsetParam {
    uint64_t mL1Size;
    uint64_t kaL1Size;
    uint64_t kbL1Size;
    uint64_t nL1Size;

    uint64_t mOffset;
    uint64_t nOffset;

    uint64_t mSize;
    uint64_t kSize;
    uint64_t nSize;
    uint64_t nAlign;

    GM_ADDR yGmAddr;
};

struct VecAntiQuantConfig {
    uint64_t ubMte2BufferNum = 2;
};
}  // namespace WeightQuantBatchMatmulV2::Arch35
#endif  // GROUPED_MATMUL_WEIGHT_QUANT_BASIC_BLOCK_CONFIG_H
