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
 * \file aggregate_hidden_grad_arch35.h
 * \brief AggregateHiddenGrad tiling implementation
 */
#ifndef AGGREGATE_HIDDEN_GRAD_TILING_ARCH35_H
#define AGGREGATE_HIDDEN_GRAD_TILING_ARCH35_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "../op_kernel/arch35/aggregate_hidden_grad_struct.h"

namespace optiling {

// CompileInfo structure for platform information (arch35)
struct AggregateHiddenGradArch35CompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

constexpr uint64_t TILING_KEY_BF16 = 10000;
constexpr uint64_t TILING_KEY_FP16 = 10001;

// Input tensor indices
constexpr int32_t GRAD_OUTPUT_INDEX = 0;
constexpr int32_t INPUT_INDEX = 1;
constexpr int32_t WEIGHT_INDEX = 2;
constexpr int32_t MASK_INDEX = 3; // optional

// Constants for validation and tiling
constexpr int64_t DIM_ALIGN_ELEMENT = 64;             // H split granularity (elements)
constexpr int64_t ALIGN_BYTES = 32;                   // Base alignment requirement
constexpr int64_t DTYPE_SIZE = 2;                     // bf16/fp16 size in bytes
constexpr int64_t BUFFER_NUM = 2;                     // Double buffering recommended
constexpr int64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024; // Reserve for system usage

// Shape dim indices
constexpr int64_t DIM_0 = 0; // S or W
constexpr int64_t DIM_1 = 1; // B or H
constexpr int64_t DIM_2 = 2; // H

class AggregateHiddenGradTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit AggregateHiddenGradTiling(gert::TilingContext *context) : TilingBaseClass(context)
    {
    }

protected:
    // Capability & info collection
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;

    // Tiling main entry
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;

    // Post processing
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    // Validation helpers
    ge::graphStatus CheckInputParams();
    ge::graphStatus ValidateGradOutputShape();
    ge::graphStatus ValidateInputShape();
    ge::graphStatus ValidateWeightShape();
    ge::graphStatus ValidateMaskShape();

    ge::graphStatus ValidateGradOutputType();
    ge::graphStatus ValidateInputType();
    ge::graphStatus ValidateWeightType();
    ge::graphStatus ValidateMaskType();

private:
    // Tiling helpers
    ge::graphStatus ComputeInterCoreSplit();    // 核间切分（H维度，非均匀）
    ge::graphStatus ComputeIntraCoreUbTiling(); // 核内切分（优先满载B、S，H从64起步）

    // Hardware information
    uint64_t ubSize_ = 0;       // UB size per core (bytes)
    uint64_t totalCoreNum_ = 0; // Total AIV cores

    // Input tensor shape information
    int64_t S_ = 0; // seq length
    int64_t B_ = 0; // batch size
    int64_t H_ = 0; // hidden size
    int64_t W_ = 0; // kernel width (must be 3)

    // Data type information
    ge::DataType dataType_{}; // f16/bf16
    size_t dtypeSize_ = DTYPE_SIZE;
    int64_t hasMask_ = 0; // 1 if mask provided

    // Inter-core tiling parameters (H non-uniform split by 64)
    int64_t hMainCoreCnt_ = 0; // big cores count
    int64_t hTailCoreCnt_ = 0; // small cores count
    int64_t hMainSize_ = 0;    // elements per big core (multiple of 64)
    int64_t hTailSize_ = 0;    // elements per small core (multiple of 64)
    int64_t usedCoreNum_ = 0;  // total used core number

    // Intra-core tiling parameters (UB loop for H/B/S)
    int64_t hUB_ = DIM_ALIGN_ELEMENT; // start with 64
    int64_t bUB_ = 1;
    int64_t sUB_ = 1;

    int64_t hLoopCnt_ = 0;
    int64_t bLoopCnt_ = 0;
    int64_t sLoopCnt_ = 0;

    int64_t hUBTail_ = 0;
    int64_t bUBTail_ = 0;
    int64_t sUBTail_ = 0;

    // TilingData object
    AggregateHiddenGradArch35Tiling::AggregateHiddenGradTilingDataV35 tilingData_{};
};

} // namespace optiling

#endif // OPS_TRANSFORMER_ATTENTION_AGGREGATE_HIDDEN_GRAD_OP_HOST_H
