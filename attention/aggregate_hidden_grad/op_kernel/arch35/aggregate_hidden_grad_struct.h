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
 * \file aggregate_hidden_grad_struct.h
 * \brief Tiling data struct for aggregate_hidden_grad on arch35
 */

#ifndef AGGREGATE_HIDDEN_GRAD_STRUCT_H
#define AGGREGATE_HIDDEN_GRAD_STRUCT_H

namespace AggregateHiddenGradArch35Tiling {

struct AggregateHiddenGradTilingDataV35 {
    // 核间切分参数
    int64_t hMainCoreCnt{0};           // h维度主核核数
    int64_t hTailCoreCnt{0};           // h维度尾核核数
    int64_t hMainSize{0};              // h维度主核处理的大小
    int64_t hTailSize{0};              // h维度尾核处理的大小

    // 主核循环参数
    int64_t hloopCnt{0};               // 主核UB内h维度循环次数
    int64_t bLoopCnt{0};               // 主核UB内b维度循环次数
    int64_t sLoopCnt{0};               // 主核UB内s维度循环次数

    // 主核UB切块参数
    int64_t ubMainFactorH{0};          // 主核UB内h维度主块大小
    int64_t ubTailFactorH{0};          // 主核UB内h维度尾块大小
    int64_t ubMainFactorB{0};          // 主核UB内b维度主块大小
    int64_t ubTailFactorB{0};          // 主核UB内b维度尾块大小
    int64_t ubMainFactorS{0};          // 主核UB内s维度主块大小
    int64_t ubTailFactorS{0};          // 主核UB内s维度尾块大小

    // 尾核循环参数
    int64_t tailHloopCnt{0};           // 尾核UB内h维度循环次数
    int64_t tailBLoopCnt{0};           // 尾核UB内b维度循环次数
    int64_t tailSLoopCnt{0};           // 尾核UB内s维度循环次数

    // 尾核UB切块参数
    int64_t tailCoreUbMainFactorH{0};  // 尾核UB内h维度主块大小
    int64_t tailCoreUbTailFactorH{0};  // 尾核UB内h维度尾块大小
    int64_t tailCoreUbMainFactorB{0};  // 尾核UB内b维度主块大小
    int64_t tailCoreUbTailFactorB{0};  // 尾核UB内b维度尾块大小
    int64_t tailCoreUbMainFactorS{0};  // 尾核UB内s维度主块大小
    int64_t tailCoreUbTailFactorS{0};  // 尾核UB内s维度尾块大小

    // 全局参数
    int64_t hasMask{0};                // 1，有mask；0，无mask
    int64_t S{0};                      // S维度大小
    int64_t B{0};                      // B维度大小
    int64_t H{0};                      // H维度大小
    int64_t W{0};                      // W维度大小
};


} // namespace AggregateHiddenGradArch35Tiling

#endif // OPS_TRANSFORMER_ATTENTION_AGGREGATE_HIDDEN_GRAD_OP_KERNEL_ARCH35_STRUCT_H
