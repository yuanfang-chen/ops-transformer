/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_IMPL_MOE_INIT_ROUTING_V3_MX_QUANT_TILING_ARCH35_H_
#define OP_IMPL_MOE_INIT_ROUTING_V3_MX_QUANT_TILING_ARCH35_H_

#include "register/tilingdata_base.h"
#include "exe_graph/runtime/tiling_context.h"
#include "graph/types.h"

namespace optiling {

// Sub-stage tiling data structures (must match op_kernel/ struct exactly)
struct MoeV3Arch35VBSComputeTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
    int64_t oneLoopMaxElements{0};
};

struct MoeV3Arch35VMSMiddleComputeTilingData {
    int64_t needCoreNum{0};
};

struct MoeV3Arch35SortOutComputeTilingData {
    int64_t oneLoopMaxElements{0};
};

struct MoeV3Arch35ExpertTokensCountTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreElements{0};
    int64_t lastCoreElements{0};
    int64_t perCoreLoops{0};
    int64_t perCorePerLoopElements{0};
    int64_t perCoreLastLoopElements{0};
    int64_t lastCoreLoops{0};
    int64_t lastCorePerLoopElements{0};
    int64_t lastCoreLastLoopElements{0};
};

struct MoeV3Arch35GatherOutComputeTilingData {
    int64_t needCoreNum{0};
    int64_t perCoreIndicesElements{0};
    int64_t lastCoreIndicesElements{0};
    int64_t perCoreIndicesLoops{0};
    int64_t perCorePerLoopIndicesElements{0};
    int64_t perCoreLastLoopIndicesElements{0};
    int64_t lastCoreIndicesLoops{0};
    int64_t lastCorePerLoopIndicesElements{0};
    int64_t lastCoreLastLoopIndicesElements{0};
    int64_t colsLoops{0};
    int64_t perLoopCols{0};
    int64_t lastLoopCols{0};
    int64_t activeNum{0};
};

// Main TilingData structure (field names, types, order must match op_kernel/_struct.h exactly)
struct MoeInitRoutingV3MxQuantArch35TilingData {
    // === Inherited from MoeInitRoutingV3 ===
    int64_t coreNum{0};
    int64_t n{0};                       // num_tokens
    int64_t cols{0};                    // hidden_size
    int64_t k{0};                       // topk
    int64_t expertStart{0};
    int64_t expertEnd{0};
    int64_t actualExpertNum{0};
    int64_t rowIdxType{0};              // 0=GATHER, 1=SCATTER
    int64_t isInputScale{0};            // whether scale input is present
    int64_t isInputOffset{0};           // whether offset input is present
    int64_t expertNum{0};
    int64_t expertTokensNumType{0};     // maps to expert_tokens_count_or_cumsum_flag attr
    int64_t expertTokensNumFlag{0};     // maps to expert_tokens_before_capacity_flag attr
    int64_t activeNum{0};               // effective token count, -1 means use all
    int64_t dropPadMode{0};

    // === MX quantization fields ===
    int64_t dstType{0};                 // 0=fp8_e4m3fn, 1=fp8_e5m2
    int64_t blocksize{0};               // MX block size, default 32
    int64_t scaleAlg{0};                // mxscale algorithm: 0=maxExp path

    // === Per-stage tiling sub-structures ===
    MoeV3Arch35VBSComputeTilingData vbsComputeParamsOp;
    MoeV3Arch35VMSMiddleComputeTilingData vmsMiddleComputeParamsOp;
    MoeV3Arch35SortOutComputeTilingData sortOutComputeParamsOp;
    MoeV3Arch35ExpertTokensCountTilingData expertTokensCountTilingDataOp;
    MoeV3Arch35GatherOutComputeTilingData gatherOutComputeParamsOp;
};

// TilingKey definitions
constexpr int64_t MOE_V3_MX_SORTONECORE_GATHER = 1030000;
constexpr int64_t MOE_V3_MX_SORTONECORE_SCATTER = 1031000;
constexpr int64_t MOE_V3_MX_SORTMULTICORE_GATHER = 1130000;
constexpr int64_t MOE_V3_MX_SORTMULTICORE_SCATTER = 1131000;

}  // namespace optiling

#endif  // OP_IMPL_MOE_INIT_ROUTING_V3_MX_QUANT_TILING_ARCH35_H_
