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
 * \file causal_conv1d_update_tiling_arch35.h
 * \brief CausalConv1dUpdate tiling implementation
 */
#ifndef CAUSAL_CONV1D_UPDATE_TILING_H
#define CAUSAL_CONV1D_UPDATE_TILING_H

#include "log/log.h"
#include "platform/platform_info.h"
#include "register/op_impl_registry.h"
#include "register/tilingdata_base.h"
#include "tiling_base/tiling_base.h"
#include "tiling_base/tiling_templates_registry.h"
#include "util/math_util.h"
#include "util/platform_util.h"
#include "util/shape_util.h"
#include "../op_kernel/arch35/causal_conv1d_update_struct.h"

namespace optiling {

// CompileInfo structure for platform information
struct CausalConv1dUpdateCompileInfo {
    uint64_t coreNum = 0;
    uint64_t ubSize = 0;
};

// Input tensor indices
constexpr int32_t X_INDEX = 0;
constexpr int32_t WEIGHT_INDEX = 1;
constexpr int32_t CONV_STATES_INDEX = 2;
constexpr int32_t QUERY_START_LOC_INDEX = 3;
constexpr int32_t CACHE_INDICES_INDEX = 4;
constexpr int32_t HAS_INITIAL_STATE_INDEX = 5;
constexpr int32_t NUM_ACCEPTED_TOKENS_INDEX = 7;

// Output tensor indices
constexpr int32_t Y_INDEX = 0;
constexpr int32_t OUTPUT_CONV_STATES_INDEX = 1;

// Attribute indices
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int32_t ATTR_RESIDUAL_CONN_MODE_INDEX = 2;
constexpr int32_t ATTR_RUN_MODE_INDEX = 3;


class CausalConv1dUpdateTiling : public Ops::Transformer::OpTiling::TilingBaseClass {
public:
    explicit CausalConv1dUpdateTiling(gert::TilingContext* context) : TilingBaseClass(context) {}

protected:
    bool IsCapable() override;
    ge::graphStatus GetShapeAttrsInfo() override;
    ge::graphStatus GetPlatformInfo() override;
    ge::graphStatus DoOpTiling() override;
    ge::graphStatus DoLibApiTiling() override;
    uint64_t GetTilingKey() const override;
    ge::graphStatus GetWorkspaceSize() override;
    ge::graphStatus PostTiling() override;
    void DumpTilingInfo() override;

    // Shape validation functions for each tensor
    ge::graphStatus ValidateXShape();
    ge::graphStatus ValidateWeightShape();
    ge::graphStatus ValidateConvStatesShape();
    ge::graphStatus ValidateQueryStartLocShape();
    ge::graphStatus ValidateCacheIndicesShape();
    ge::graphStatus ValidateNumAcceptedTokensShape();

    // Type validation functions for each tensor
    ge::graphStatus ValidateXType();
    ge::graphStatus ValidateWeightType();
    ge::graphStatus ValidateConvStatesType();
    ge::graphStatus ValidateQueryStartLocType();
    ge::graphStatus ValidateCacheIndicesType();
    ge::graphStatus ValidateNumAcceptedTokensType();

    // Overall validation
    ge::graphStatus CheckInputParams();

private:
    // Tiling calculation functions
    int64_t CalculateLimitedCoreNum();
    int64_t ComputeOptimalDimChunk(int64_t dim, int64_t batch, int64_t coreNum);
    void CalculateTilingParams(int64_t validBatch);
    void CalculateIntraCoreTiling();

    // Hardware information
    uint64_t ubSize_ = 0;
    uint64_t totalCoreNum_ = 0;
    uint64_t ubBlockSize_ = 0;

    // Input tensor shape information
    int64_t batchSize_ = 0;
    int64_t seqLen_ = 0;
    int64_t cuSeqLen_ = 0;  // For 2D input: first dimension of x, equals batch * seq_len
    int64_t dim_ = 0;
    int64_t kernelSize_ = 0;

    // Data type information
    ge::DataType xDtype_;
    ge::DataType weightDtype_;
    ge::DataType convStatesDtype_;
    ge::DataType queryStartLocDtype_;
    ge::DataType cacheIndicesDtype_;
    ge::DataType numAcceptedTokensDtype_;
    size_t xDtypeSize_ = 0;

    // Attribute values
    int64_t padSlotId_ = -1;
    int64_t residualConnMode_ = 0;
    int64_t runMode_ = 0;
    int64_t inValidBatchNum_ = 0;
    int64_t xInputMode_ = 0;  // 0 for 3D [batch, seq_len, dim], 1 for 2D [cu_seq_len, dim]
    int64_t hasAcceptTokenNum_ = 0;  // Whether acceptTokenNum input is provided: 0 for false, 1 for true

    // Tiling parameters
    int64_t limitedCoreNum_ = 0;      // Limited core number based on data size
    int64_t usedCoreNum_ = 0;         // Actually used core number
    int64_t dimCoreCnt_ = 0;          // Number of cores for dim direction
    int64_t batchCoreCnt_ = 0;        // Number of cores for batch direction
    int64_t dimChunkSize_ = 0;        // Dim chunk size per core (256 * N)
    int64_t dimTailSize_ = 0;         // Dim tail size for last core
    int64_t batchPerCore_ = 0;        // Batches per core (regular)
    int64_t batchTailPerCore_ = 0;    // Batches for tail core
    int64_t validBatchStart_ = 0;     // First valid batch index
    int64_t validBatchEnd_ = 0;       // Last valid batch index (inclusive)

    // Intra-core tiling parameters
    int64_t ubBatchSize_ = 0;         // Batch size per UB iteration
    int64_t ubDimSize_ = 0;           // Dim size per UB iteration (elements)
    int64_t batchLoopCnt_ = 0;        // Batch loop count within core
    int64_t dimLoopCnt_ = 0;          // Dim loop count within core

    // TilingData object
    CausalConv1dUpdateTilingData tilingData_;
};

} // namespace optiling

#endif // CAUSAL_CONV1D_UPDATE_TILING_H
