/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*!\n * \file aggregate_hidden_grad_struct.h\n * \brief Tiling data struct for aggregate_hidden_grad on arch35\n */
#ifndef OPS_TRANSFORMER_ATTENTION_AGGREGATE_HIDDEN_GRAD_OP_KERNEL_ARCH35_STRUCT_H
#define OPS_TRANSFORMER_ATTENTION_AGGREGATE_HIDDEN_GRAD_OP_KERNEL_ARCH35_STRUCT_H


namespace AggregateHiddenGradArch35Tiling {


struct AggregateHiddenGradTilingDataV35 {
    // global flags and shapes
    int64_t hasMask{0};
    int64_t H{0};
    int64_t S{0};
    int64_t B{0};
    int64_t W{0};         // must be 3
    int64_t dtypeSize{0}; // bytes per element of DT

    // inter-core split on H
    int64_t hMainCoreCnt{0};
    int64_t hTailCoreCnt{0};
    int64_t hMainSize{0}; // per-core H size for main cores
    int64_t hTailSize{0}; // per-core H size for tail cores

    // intra-core UB tiling sizes
    int64_t hUB{0};
    int64_t bUB{0};
    int64_t sUB{0};

    // loop counts for main block
    int64_t hLoopCnt{0};
    int64_t bLoopCnt{0};
    int64_t sLoopCnt{0};

    // tail sizes and loop counts for the last tile in each dim
    int64_t hUBTail{0};
    int64_t bUBTail{0};
    int64_t sUBTail{0};

    int64_t hLoopCntTail{0};
    int64_t bLoopCntTail{0};
    int64_t sLoopCntTail{0};

    // misc
    int64_t coreMainRangeStart{0}; // base H offset for the first main-core range
    int64_t alignBytes{32};        // 32B alignment
};


} // namespace AggregateHiddenGradArch35Tiling

#endif // OPS_TRANSFORMER_ATTENTION_AGGREGATE_HIDDEN_GRAD_OP_KERNEL_ARCH35_STRUCT_H
