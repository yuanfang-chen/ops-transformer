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
 * \file causal_conv1d_update_tiling.h
 * \brief CausalConv1dUpdate tiling implementation
 */
#ifndef CAUSAL_CONV1D_UPDATE_TILING_H
#define CAUSAL_CONV1D_UPDATE_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"

namespace optiling {

// Input tensor indices
constexpr int32_t X_INDEX = 0;
constexpr int32_t FILTER_INDEX = 1;
constexpr int32_t CACHE_STATE_INDEX = 2;
constexpr int32_t CACHE_INDICES_INDEX = 3;
constexpr int32_t ACCEPT_TOKEN_NUM_INDEX = 4;

// Output tensor indices
constexpr int32_t Y_INDEX = 0;
constexpr int32_t OUTPUT_CACHE_STATE_INDEX = 1;

// Attribute indices
constexpr int32_t ATTR_PAD_SLOT_INDEX_INDEX = 0;

// TilingData structure definition
BEGIN_TILING_DATA_DEF(CausalConv1dUpdateTilingData)
TILING_DATA_FIELD_DEF(int64_t, blockFactor);              // Block factor for core distribution
TILING_DATA_FIELD_DEF(int64_t, blockTailFactor);          // Tail block factor
TILING_DATA_FIELD_DEF(int64_t, loopNumBS);                // Number of loops in BS direction
TILING_DATA_FIELD_DEF(int64_t, loopNumDim);               // Number of loops in Dim direction
TILING_DATA_FIELD_DEF(int64_t, ubFactorBS);               // UB factor for BS per loop
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorBS);           // UB tail factor for BS
TILING_DATA_FIELD_DEF(int64_t, ubFactorDim);              // UB factor for Dim per loop
TILING_DATA_FIELD_DEF(int64_t, ubTailFactorDim);          // UB tail factor for Dim
TILING_DATA_FIELD_DEF(int64_t, tailBlockloopNumBS);       // Tail block loop count for BS
TILING_DATA_FIELD_DEF(int64_t, tailBlockloopNumDim);      // Tail block loop count for Dim
TILING_DATA_FIELD_DEF(int64_t, tailBlockubFactorBS);      // Tail block UB factor for BS
TILING_DATA_FIELD_DEF(int64_t, tailBlockubTailFactorBS);  // Tail block UB tail factor for BS
TILING_DATA_FIELD_DEF(int64_t, tailBlockubFactorDim);     // Tail block UB factor for Dim
TILING_DATA_FIELD_DEF(int64_t, tailBlockubTailFactorDim); // Tail block UB tail factor for Dim
TILING_DATA_FIELD_DEF(int64_t, batchSize);                // Batch size
TILING_DATA_FIELD_DEF(int64_t, seqLen);                   // Sequence length
TILING_DATA_FIELD_DEF(int64_t, dim);                      // Dimension size
TILING_DATA_FIELD_DEF(int64_t, kernelSize);               // Kernel size (K)
END_TILING_DATA_DEF;

REGISTER_TILING_DATA_CLASS(CausalConv1dUpdate, CausalConv1dUpdateTilingData)

class CausalConv1dUpdateTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit CausalConv1dUpdateTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    // Shape validation functions for each tensor
    ge::graphStatus ValidateXShape();
    ge::graphStatus ValidateFilterShape();
    ge::graphStatus ValidateCacheStateShape();
    ge::graphStatus ValidateCacheIndicesShape();
    ge::graphStatus ValidateAcceptTokenNumShape();

    // Type validation functions for each tensor
    ge::graphStatus ValidateXType();
    ge::graphStatus ValidateFilterType();
    ge::graphStatus ValidateCacheStateType();
    ge::graphStatus ValidateCacheIndicesType();
    ge::graphStatus ValidateAcceptTokenNumType();

    // Overall validation
    ge::graphStatus CheckInputParams();

private:
    // Tiling calculation functions
    void CalculateCoreParams(int64_t validBatch);
    void CalculateLoopParams(bool isTailBlock);

    // Hardware information
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t ubBlockSize_ = 0;

    // Input tensor shape information
    int64_t batchSize_ = 0;
    int64_t seqLen_ = 0;
    int64_t dim_ = 0;
    int64_t kernelSize_ = 0;

    // Data type information
    ge::DataType xDtype_;
    ge::DataType filterDtype_;
    ge::DataType cacheStateDtype_;
    ge::DataType cacheIndicesDtype_;
    ge::DataType acceptTokenNumDtype_;
    size_t xDtypeSize_ = 0;

    // Attribute values
    int64_t padSlotIndex_ = 0;
    int64_t inValidBatchNum_ = 0;

    // Tiling parameters
    int64_t usedCoreNum_ = 0;
    int64_t blockFactor_ = 0;
    int64_t blockTailFactor_ = 0;

    // Loop parameters for regular block
    int64_t loopNumBS_ = 0;
    int64_t loopNumDim_ = 0;
    int64_t ubFactorBS_ = 0;
    int64_t ubTailFactorBS_ = 0;
    int64_t ubFactorDim_ = 0;
    int64_t ubTailFactorDim_ = 0;

    // Loop parameters for tail block
    int64_t tailBlockloopNumBS_ = 0;
    int64_t tailBlockloopNumDim_ = 0;
    int64_t tailBlockubFactorBS_ = 0;
    int64_t tailBlockubTailFactorBS_ = 0;
    int64_t tailBlockubFactorDim_ = 0;
    int64_t tailBlockubTailFactorDim_ = 0;

    // TilingData object
    CausalConv1dUpdateTilingData tilingData_;
};

} // namespace optiling

#endif // CAUSAL_CONV1D_UPDATE_TILING_H
