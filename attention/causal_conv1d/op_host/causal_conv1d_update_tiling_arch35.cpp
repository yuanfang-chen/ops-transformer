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
 * \file causal_conv1d_update_tiling_arch35.cpp
 * \brief CausalConv1dUpdate tiling implementation
 */

#include "causal_conv1d_update_tiling_arch35.h"
#include <algorithm>
#include "securec.h"

namespace optiling {


#define TILING_KEY_UPDATE_BF16 20000
// #define TILING_KEY_UPDATE_FP16 20001

bool CausalConv1dUpdateTiling::IsCapable()
{
    return true;
}

ge::graphStatus CausalConv1dUpdateTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const CausalConv1dUpdateCompileInfo *>(context_->GetCompileInfo());
        OP_CHECK_IF(compileInfoPtr == nullptr, OP_LOGE(context_, "compile info is null"), return ge::GRAPH_FAILED);
        totalCoreNum_ = compileInfoPtr->coreNum;
        ubSize_ = compileInfoPtr->ubSize;
    } else {
        auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
        totalCoreNum_ = static_cast<uint64_t>(ascendcPlatform.GetCoreNumAiv());
        if (totalCoreNum_ == 0UL) {
            OP_LOGE(context_->GetNodeName(), "coreNum is 0");
            return ge::GRAPH_FAILED;
        }
        uint64_t ubSize = 0;
        ascendcPlatform.GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubSize);
        if (ubSize == static_cast<uint64_t>(0)) {
            OP_LOGE(context_->GetNodeName(), "ubSize is 0");
            return ge::GRAPH_FAILED;
        }
        ubSize_ = static_cast<uint64_t>(ubSize);
    }

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdateTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("CausalConv1dUpdate", "context is null"),
                return ge::GRAPH_FAILED);

    // Get x shape
    auto xShape = context_->GetInputShape(X_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, xShape);
    auto xOriginShape = xShape->GetOriginShape();

    // Support both 3D [batch, seq_len, dim] and 2D [cu_seq_len, dim] input
    if (xOriginShape.GetDimNum() == DIM_3) {
        xInputMode_ = X_INPUT_3D;  // 3D input mode
        batchSize_ = xOriginShape.GetDim(DIM_0);
        seqLen_ = xOriginShape.GetDim(DIM_1);
        dim_ = xOriginShape.GetDim(DIM_2);
    } else if (xOriginShape.GetDimNum() == DIM_2) {
        xInputMode_ = X_INPUT_2D;  // 2D input mode
        cuSeqLen_ = xOriginShape.GetDim(DIM_0);  // cu_seq_len = batch * seq_len
        dim_ = xOriginShape.GetDim(DIM_1);

        // For 2D input, query_start_loc must be specified to get batch
        auto queryStartLocShape = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, queryStartLocShape);
        auto queryStartLocOriginShape = queryStartLocShape->GetOriginShape();

        // query_start_loc shape is (batch + 1,), so batch = dim0 - 1
        batchSize_ = queryStartLocOriginShape.GetDim(DIM_0) - 1;
    } else {
        OP_LOGE(context_->GetNodeName(), "X dimension number must be 2 or 3, but got %lu",
                xOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    // Get weight shape
    auto weightShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    auto weightOriginShape = weightShape->GetOriginShape();

    if (weightOriginShape.GetDimNum() != DIM_2) {
        OP_LOGE(context_->GetNodeName(), "Weight dimension number must be 2, but got %lu",
                weightOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    kernelSize_ = weightOriginShape.GetDim(DIM_0);

    // Get data types
    xDtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    weightDtype_ = context_->GetInputDesc(WEIGHT_INDEX)->GetDataType();
    convStatesDtype_ = context_->GetInputDesc(CONV_STATES_INDEX)->GetDataType();

    // Get queryStartLoc dtype if available (optional input)
    auto queryStartLocDesc = context_->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    if (queryStartLocDesc != nullptr) {
        queryStartLocDtype_ = queryStartLocDesc->GetDataType();
    }

    // Get cacheIndices dtype if available (optional input)
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    if (cacheIndicesDesc != nullptr) {
        cacheIndicesDtype_ = cacheIndicesDesc->GetDataType();
    }

    // Get numAcceptedToken dtype if available (optional input)
    auto numAcceptedTokenDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKEN_INDEX);
    if (numAcceptedTokenDesc != nullptr) {
        numAcceptedTokenDtype_ = numAcceptedTokenDesc->GetDataType();
        hasAcceptTokenNum_ = 1;  // true
    } else {
        hasAcceptTokenNum_ = 0;  // false
    }

    // Get dtype size
    xDtypeSize_ = GetSizeByDataType(xDtype_);
    OP_CHECK_IF(xDtypeSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dUpdate get x dtype[%s] size is 0.",
                        Ops::Base::ToString(xDtype_).c_str()),
                return ge::GRAPH_FAILED);

    // Get attributes
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t* activationModePtr = attrs->GetAttrPointer<int64_t>(ATTR_ACTIVATION_MODE_INDEX);
    if (activationModePtr != nullptr) {
        activationMode_ = *activationModePtr;
    }

    const int64_t* padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    if (padSlotIdPtr != nullptr) {
        padSlotId_ = *padSlotIdPtr;
    }

    const int64_t* runModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RUN_MODE_INDEX);
    if (runModePtr != nullptr) {
        runMode_ = *runModePtr;
    }

    // Get convStates shape to retrieve stateLen
    auto convStatesShape = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesShape);
    auto convStatesOriginShape = convStatesShape->GetOriginShape();
    // stateLen is the second dimension of convStates [-1, stateLen, dim]
    stateLen_ = convStatesOriginShape.GetDim(DIM_1);

    // Perform all validations
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dUpdate CheckInputParams FAILED."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate X tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateXShape()
{
    // Validate batch size: [1, 256]
    OP_CHECK_IF(batchSize_ < MIN_BATCH || batchSize_ > MAX_BATCH,
                OP_LOGE(context_->GetNodeName(),
                        "X batch size must be in [%ld, %ld], but got %ld",
                        MIN_BATCH, MAX_BATCH, batchSize_),
                return ge::GRAPH_FAILED);

    // For 3D input, validate sequence length
    if (xInputMode_ == X_INPUT_3D) {
        // Validate sequence length: m+1 where m in [0, 5], so seqLen in [1, 6]
        int64_t m = seqLen_ - 1;
        OP_CHECK_IF(m < MIN_M || m > MAX_M,
                    OP_LOGE(context_->GetNodeName(),
                            "X sequence length must be m+1 where m in [%ld, %ld], but got seqLen=%ld (m=%ld)",
                            MIN_M, MAX_M, seqLen_, m),
                    return ge::GRAPH_FAILED);
    }
    // For 2D input, cuSeqLen_ is used instead of seqLen_

    // Validate dimension: [64, 16384]
    OP_CHECK_IF(dim_ < MIN_DIM || dim_ > MAX_DIM,
                OP_LOGE(context_->GetNodeName(),
                        "X dimension must be in [%ld, %ld], but got %ld",
                        MIN_DIM, MAX_DIM, dim_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate weight tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateWeightShape()
{
    auto weightShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    auto weightOriginShape = weightShape->GetOriginShape();

    // Validate weight dim matches x dim
    int64_t weightDim = weightOriginShape.GetDim(1);
    OP_CHECK_IF(weightDim != dim_,
                OP_LOGE(context_->GetNodeName(),
                        "Weight dimension must match X dimension %ld, but got %ld",
                        dim_, weightDim),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate conv states tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateConvStatesShape()
{
    auto convStatesShape = context_->GetInputShape(CONV_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, convStatesShape);
    auto convStatesOriginShape = convStatesShape->GetOriginShape();

    // Validate dimension number
    OP_CHECK_IF(convStatesOriginShape.GetDimNum() != DIM_3,
                OP_LOGE(context_->GetNodeName(),
                        "ConvStates dimension number must be 3, but got %lu",
                        convStatesOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // conv states shape: [-1, K-1+m, dim]
    // The second dimension should be K-1 + m = K-1 + (seqLen-1) = K + seqLen - 2
    // state_len must be greater than the maximum of width-1+seq_len-1 for all batches. 
    int64_t expectedCacheLen = kernelSize_ + seqLen_ - 2;
    int64_t state_len = convStatesOriginShape.GetDim(DIM_1);
    OP_CHECK_IF(state_len < expectedCacheLen,
                OP_LOGE(context_->GetNodeName(),
                        "state_len must be greater than width-1+seq_len-1 = %ld, but got %ld",
                        expectedCacheLen, state_len),
                return ge::GRAPH_FAILED);

    // Validate conv states dim matches x dim
    int64_t convStatesDim = convStatesOriginShape.GetDim(DIM_2);
    OP_CHECK_IF(convStatesDim != dim_,
                OP_LOGE(context_->GetNodeName(),
                        "ConvStates dimension must match X dimension %ld, but got %ld",
                        dim_, convStatesDim),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate cache indices tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateCacheIndicesShape()
{
    // This is an optional input
    auto indicesShape = context_->GetOptionalInputShape(CACHE_INDICES_INDEX);
    if (indicesShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto indicesOriginShape = indicesShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(indicesOriginShape.GetDimNum() != DIM_1,
                OP_LOGE(context_->GetNodeName(),
                        "CacheIndices dimension number must be 1, but got %lu",
                        indicesOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate shape matches batch size
    int64_t indicesLen = indicesOriginShape.GetDim(DIM_0);
    OP_CHECK_IF(indicesLen != batchSize_,
                OP_LOGE(context_->GetNodeName(),
                        "CacheIndices length must match batch size %ld, but got %ld",
                        batchSize_, indicesLen),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate accept token num tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateNumAcceptedTokenShape()
{
    // This is an optional input
    if (context_->GetOptionalInputTensor(NUM_ACCEPTED_TOKEN_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    auto acceptShape = context_->GetOptionalInputShape(NUM_ACCEPTED_TOKEN_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, acceptShape);
    auto acceptOriginShape = acceptShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(acceptOriginShape.GetDimNum() != DIM_1,
                OP_LOGE(context_->GetNodeName(),
                        "NumAcceptedToken dimension number must be 1, but got %lu",
                        acceptOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate shape matches batch size
    int64_t acceptLen = acceptOriginShape.GetDim(DIM_0);
    OP_CHECK_IF(acceptLen != batchSize_,
                OP_LOGE(context_->GetNodeName(),
                        "NumAcceptedToken length must match batch size %ld, but got %ld",
                        batchSize_, acceptLen),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate query start loc tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateQueryStartLocShape()
{
    // This is an optional input
    auto queryStartLocShape = context_->GetOptionalInputShape(QUERY_START_LOC_INDEX);
    if (queryStartLocShape == nullptr) {
        return ge::GRAPH_SUCCESS;
    }
    auto queryStartLocOriginShape = queryStartLocShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(queryStartLocOriginShape.GetDimNum() != DIM_1,
                OP_LOGE(context_->GetNodeName(),
                        "QueryStartLoc dimension number must be 1, but got %lu",
                        queryStartLocOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate shape: should be (batch + 1,)
    int64_t queryStartLocLen = queryStartLocOriginShape.GetDim(DIM_0);
    OP_CHECK_IF(queryStartLocLen != batchSize_ + 1,
                OP_LOGE(context_->GetNodeName(),
                        "QueryStartLoc length must be batch_size + 1 = %ld, but got %ld",
                        batchSize_ + 1, queryStartLocLen),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate X tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateXType()
{
    OP_CHECK_IF(xDtype_ != ge::DataType::DT_FLOAT16 && xDtype_ != ge::DataType::DT_BF16,
                OP_LOGE(context_->GetNodeName(),
                        "X data type must be FLOAT16 or BFLOAT16, but got %s",
                        Ops::Base::ToString(xDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate weight tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateWeightType()
{
    OP_CHECK_IF(weightDtype_ != xDtype_,
                OP_LOGE(context_->GetNodeName(),
                        "Weight data type must match X data type %s, but got %s",
                        Ops::Base::ToString(xDtype_).c_str(),
                        Ops::Base::ToString(weightDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate conv states tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateConvStatesType()
{
    OP_CHECK_IF(convStatesDtype_ != xDtype_,
                OP_LOGE(context_->GetNodeName(),
                        "ConvStates data type must match X data type %s, but got %s",
                        Ops::Base::ToString(xDtype_).c_str(),
                        Ops::Base::ToString(convStatesDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate cache indices tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateCacheIndicesType()
{
    // This is an optional input
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(CACHE_INDICES_INDEX);
    if (cacheIndicesDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(cacheIndicesDtype_ != ge::DataType::DT_INT32,
                OP_LOGE(context_->GetNodeName(),
                        "CacheIndices data type must be INT32, but got %s",
                        Ops::Base::ToString(cacheIndicesDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate query start loc tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateQueryStartLocType()
{
    // This is an optional input
    auto queryStartLocDesc = context_->GetOptionalInputDesc(QUERY_START_LOC_INDEX);
    if (queryStartLocDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(queryStartLocDtype_ != ge::DataType::DT_INT32,
                OP_LOGE(context_->GetNodeName(),
                        "QueryStartLoc data type must be INT32, but got %s",
                        Ops::Base::ToString(queryStartLocDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate num accepted tokens tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateNumAcceptedTokenType()
{
    // This is an optional input
    auto numAcceptedTokenDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKEN_INDEX);
    if (numAcceptedTokenDesc == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(numAcceptedTokenDtype_ != ge::DataType::DT_INT32,
                OP_LOGE(context_->GetNodeName(),
                        "NumAcceptedToken data type must be INT32, but got %s",
                        Ops::Base::ToString(numAcceptedTokenDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Overall input parameters validation
ge::graphStatus CausalConv1dUpdateTiling::CheckInputParams()
{
    // Validate all shapes
    OP_CHECK_IF(ValidateXShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "X shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateWeightShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Weight shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateConvStatesShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ConvStates shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateQueryStartLocShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "QueryStartLoc shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateCacheIndicesShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CacheIndices shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateNumAcceptedTokenShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "NumAcceptedToken shape validation failed"),
                return ge::GRAPH_FAILED);

    // Validate all types
    OP_CHECK_IF(ValidateXType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "X type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateWeightType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Weight type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateConvStatesType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "ConvStates type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateQueryStartLocType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "QueryStartLoc type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateCacheIndicesType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CacheIndices type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateNumAcceptedTokenType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "NumAcceptedToken type validation failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdateTiling::DoOpTiling()
{
    // Calculate invalid batch number by checking padSlotId
    // Invalid batches can be at the beginning or at the end
    int64_t invalidBatchAtStart = 0;
    int64_t invalidBatchAtEnd = 0;

    auto cacheIndicesTensor = context_->GetOptionalInputTensor(CACHE_INDICES_INDEX);
    if (cacheIndicesTensor != nullptr) {
        const int32_t* dataPtr = cacheIndicesTensor->GetData<int32_t>();
        if (dataPtr != nullptr) {
            // Count invalid batches at the beginning
            for (int64_t i = 0; i < batchSize_; i++) {
                if (padSlotId_ == static_cast<int64_t>(dataPtr[i])) {
                    invalidBatchAtStart++;
                } else {
                    break;
                }
            }

            // Count invalid batches at the end (avoid double counting)
            for (int64_t i = batchSize_ - 1; i >= invalidBatchAtStart; i--) {
                if (padSlotId_ == static_cast<int64_t>(dataPtr[i])) {
                    invalidBatchAtEnd++;
                } else {
                    break;
                }
            }
        }
    }

    inValidBatchNum_ = invalidBatchAtStart + invalidBatchAtEnd;

    // Record first and last valid batch indices for kernel use
    validBatchStart_ = invalidBatchAtStart;
    validBatchEnd_ = batchSize_ - 1 - invalidBatchAtEnd;

    // Calculate valid batch
    int64_t validBatch = batchSize_ - inValidBatchNum_;
    OP_CHECK_IF(validBatch <= 0,
                OP_LOGE(context_->GetNodeName(), "Valid batch must be positive, but got %ld", validBatch),
                return ge::GRAPH_FAILED);

    // Calculate tiling parameters
    CalculateTilingParams(validBatch);

    // Calculate intra-core tiling parameters (UB loop)
    CalculateIntraCoreTiling();

    return ge::GRAPH_SUCCESS;
}

int64_t CausalConv1dUpdateTiling::CalculateLimitedCoreNum()
{
    // Calculate input x size in bytes (data type is float16/bf16, 2 bytes per element)
    int64_t xSizeBytes;
    if (xInputMode_ == X_INPUT_3D) {
        // 3D input: batchSize * seqLen * dim * 2 bytes
        xSizeBytes = batchSize_ * seqLen_ * dim_ * DTYPE_SIZE;
    } else {
        // 2D input: cuSeqLen * dim * 2 bytes
        xSizeBytes = cuSeqLen_ * dim_ * DTYPE_SIZE;
    }

    // Fixed UB usage for auxiliary tensors
    // cacheIndices: batchSize_ * sizeof(int32)
    // numAcceptedToken: batchSize_ * sizeof(int32)
    // queryStartLoc: (batchSize_ + 1) * sizeof(int32)
    int64_t cacheIndicesUBSize = batchSize_ * sizeof(int32_t);
    int64_t numAcceptedTokensUBSize = batchSize_ * sizeof(int32_t);

    // For 3D input (xInputMode_ == X_INPUT_3D), only include cacheIndicesUBSize and numAcceptedTokensUBSize
    fixedUBSize = cacheIndicesUBSize + numAcceptedTokensUBSize;
    // For 2D input (xInputMode_ == X_INPUT_2D), include queryStartLocUBSize
    if (xInputMode_ == X_INPUT_2D) {
        int64_t queryStartLocUBSize = (batchSize_ + 1) * sizeof(int32_t);
        fixedUBSize += queryStartLocUBSize;
    }

    // Limit core number based on data size
    // When data occupies half of UB, bandwidth is good
    // effectiveCoreNum = xSizeBytes / (ubSize_ / 2)
    int64_t halfUbSize = (ubSize_ - fixedUBSize) / 2;
    int64_t effectiveCoreNum = (xSizeBytes + halfUbSize - 1) / halfUbSize;
    effectiveCoreNum = std::max(effectiveCoreNum, static_cast<int64_t>(1));

    // Actual core number is min(effectiveCoreNum, totalCoreNum_)
    return std::min(effectiveCoreNum, static_cast<int64_t>(totalCoreNum_));
}

int64_t CausalConv1dUpdateTiling::ComputeOptimalDimChunk(int64_t dim, int64_t batch, int64_t coreNum)
{
    /**
     * Enumerate dim chunk size = 256 * N for each core.
     *
     * - If dim <= 256*N, no need to split dim;
     * - Otherwise, need multiple cores, first K cores process 256*N, last core processes tail (<256*N);
     *
     * Max dim per core =
     *     dim,           if dimCoreCnt == 1
     *     256 * N,       if dimCoreCnt >= 2
     *
     * Goal: minimize workload = maxDimPerCore * ceil(batch / batchCnt)
     */
    if (batch <= 0 || coreNum <= 0 || dim <= 0) {
        return 1;
    }

    int64_t bestWorkload = INT64_MAX;
    int64_t bestN = 1;

    // N range is [1, ceil(dim / DIM_ALIGN_ELEMENT)]
    // Since dim max is 16384, N max is 16384 / 128 = 128
    int64_t maxN = (dim + DIM_ALIGN_ELEMENT - 1) / DIM_ALIGN_ELEMENT;

    for (int64_t N = 1; N <= maxN; N++) {
        int64_t chunkDim = DIM_ALIGN_ELEMENT * N;

        // Number of cores for dim direction
        int64_t dimCoreCnt = (dim + chunkDim - 1) / chunkDim;

        if (dimCoreCnt > coreNum) {
            continue;
        }

        // Number of cores for batch direction
        int64_t batchCoreTotal = coreNum / dimCoreCnt;
        if (batchCoreTotal == 0) {
            continue;
        }

        int64_t batchCnt = std::min(batchCoreTotal, batch);
        // Batches per core
        int64_t batchPerCore = (batch + batchCnt - 1) / batchCnt;

        int64_t maxDimPerCore;
        if (dimCoreCnt == 1) {
            maxDimPerCore = dim;
        } else {
            maxDimPerCore = chunkDim;  // tail < chunkDim, so max is chunkDim
        }

        int64_t workload = maxDimPerCore * batchPerCore;

        // Record N when workload is minimum
        if (workload < bestWorkload) {
            bestWorkload = workload;
            bestN = N;
        }
    }

    return bestN;
}

void CausalConv1dUpdateTiling::CalculateTilingParams(int64_t validBatch)
{
    // Step 1: Calculate limited core number
    limitedCoreNum_ = CalculateLimitedCoreNum();

    // Step 2: Find optimal dim chunk size
    int64_t optimalN = ComputeOptimalDimChunk(dim_, validBatch, limitedCoreNum_);
    dimChunkSize_ = DIM_ALIGN_ELEMENT * optimalN;

    // Step 3: Calculate dim core count and tail size
    dimCoreCnt_ = (dim_ + dimChunkSize_ - 1) / dimChunkSize_;
    dimTailSize_ = dim_ - (dimCoreCnt_ - 1) * dimChunkSize_;

    // Step 4: Calculate batch core count
    int64_t batchCoreTotal = limitedCoreNum_ / dimCoreCnt_;
    batchCoreCnt_ = std::min(batchCoreTotal, validBatch);
    batchCoreCnt_ = std::max(batchCoreCnt_, static_cast<int64_t>(1));

    // Step 5: Calculate batches per core
    batchPerCore_ = (validBatch + batchCoreCnt_ - 1) / batchCoreCnt_;
    batchTailPerCore_ = validBatch - (batchCoreCnt_ - 1) * batchPerCore_;

    // Step 6: Calculate total used core number
    usedCoreNum_ = dimCoreCnt_ * batchCoreCnt_;
}

void CausalConv1dUpdateTiling::CalculateIntraCoreTiling()
{
    // Use dimChunkSize_ and batchPerCore_ as the maximum data per core
    int64_t coreDim = dimChunkSize_;
    int64_t coreBatch = batchPerCore_;

    // Coefficient per dim element for weight and convStates (fixed per UB iteration)
    // weight: kernelSize_ * ubDim * DTYPE_SIZE
    // convStates: (kernelSize_ + m - 1) * ubDim * DTYPE_SIZE, where m = seqLen_ - 1
    //           = (kernelSize_ + seqLen_ - 2) * ubDim * DTYPE_SIZE
    int64_t weightConvStatesCoeffPerDim = (kernelSize_ + kernelSize_ + seqLen_ - 2) * DTYPE_SIZE;

    // x coefficient per dim element when full batch is loaded
    // x: BUFFER_NUM * coreBatch * seqLen_ * ubDim * DTYPE_SIZE
    int64_t xCoeffPerDimFullBatch = BUFFER_NUM * coreBatch * seqLen_ * DTYPE_SIZE;

    // Total coefficient per dim element with full batch
    int64_t totalCoeffPerDim = weightConvStatesCoeffPerDim + xCoeffPerDimFullBatch;

    // Calculate maximum ubDim when full batch is loaded
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - fixedUBSize;
    int64_t maxUbDim = availableUbSize / totalCoeffPerDim;

    // Align to DIM_ALIGN_ELEMENT (256 bytes = 128 bf16 elements)
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENT) * DIM_ALIGN_ELEMENT;

    if (maxUbDim >= DIM_ALIGN_ELEMENT) {
        // Can load full batch, try to maximize dim
        ubBatchSize_ = coreBatch;
        ubDimSize_ = std::min(maxUbDim, coreDim);

        // Ensure ubDimSize_ is aligned to DIM_ALIGN_ELEMENT
        ubDimSize_ = (ubDimSize_ / DIM_ALIGN_ELEMENT) * DIM_ALIGN_ELEMENT;
        if (ubDimSize_ == 0) {
            ubDimSize_ = DIM_ALIGN_ELEMENT;
        }
    } else {
        // Cannot load full batch with minimum dim, need to reduce batch
        // Use minimum ubDim = DIM_ALIGN_ELEMENT
        ubDimSize_ = DIM_ALIGN_ELEMENT;

        // Calculate space for weight and convStates with minimum dim
        int64_t weightConvStatesSize = weightConvStatesCoeffPerDim * ubDimSize_;
        int64_t availableForX = availableUbSize - weightConvStatesSize;

        // x size per batch = seqLen_ * ubDimSize_ * DTYPE_SIZE
        int64_t xSizePerBatch = seqLen_ * ubDimSize_ * DTYPE_SIZE;

        // Calculate how many batches can fit
        ubBatchSize_ = availableForX / xSizePerBatch;
        ubBatchSize_ = std::max(ubBatchSize_, static_cast<int64_t>(1));
        ubBatchSize_ = std::min(ubBatchSize_, coreBatch);
    }

    // Calculate loop counts
    // Note: one batch cannot be split across UB loops (batch has multiple seqs)
    dimLoopCnt_ = (coreDim + ubDimSize_ - 1) / ubDimSize_;
    batchLoopCnt_ = (coreBatch + ubBatchSize_ - 1) / ubBatchSize_;
}

uint64_t CausalConv1dUpdateTiling::GetTilingKey() const
{
    return TILING_KEY_UPDATE_BF16;
}

ge::graphStatus CausalConv1dUpdateTiling::PostTiling()
{
    // Set block dimension (number of cores to use)
    context_->SetBlockDim(usedCoreNum_);

    // Populate tiling data - core distribution
    tilingData_.usedCoreNum = usedCoreNum_;
    tilingData_.dimCoreCnt = dimCoreCnt_;
    tilingData_.batchCoreCnt = batchCoreCnt_;

    // Dim tiling parameters
    tilingData_.dimChunkSize = dimChunkSize_;
    tilingData_.dimTailSize = dimTailSize_;

    // Batch tiling parameters
    tilingData_.batchPerCore = batchPerCore_;
    tilingData_.batchTailPerCore = batchTailPerCore_;
    tilingData_.validBatchStart = validBatchStart_;
    tilingData_.validBatchEnd = validBatchEnd_;

    // Intra-core tiling parameters (UB loop)
    tilingData_.ubBatchSize = ubBatchSize_;
    tilingData_.ubDimSize = ubDimSize_;
    tilingData_.batchLoopCnt = batchLoopCnt_;
    tilingData_.dimLoopCnt = dimLoopCnt_;

    // Shape information for kernel use
    tilingData_.batchSize = batchSize_;
    tilingData_.seqLen = seqLen_;
    tilingData_.cuSeqLen = cuSeqLen_;
    tilingData_.dim = dim_;
    tilingData_.kernelSize = kernelSize_;
    tilingData_.stateLen = stateLen_;
    tilingData_.xStride = 0;
    tilingData_.cacheStride = 0;
    tilingData_.xInputMode = xInputMode_;
    tilingData_.hasAcceptTokenNum = hasAcceptTokenNum_;

    // Save tiling data to buffer
    auto tilingDataSize = sizeof(CausalConv1dUpdateTilingData);
    errno_t ret = memcpy_s(context_->GetRawTilingData()->GetData(),
                           context_->GetRawTilingData()->GetCapacity(),
                           reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void CausalConv1dUpdateTiling::DumpTilingInfo()
{
    OP_LOGI(context_->GetNodeName(), "=== CausalConv1dUpdate DumpTilingInfo ===");

    // Core distribution parameters
    OP_LOGI(context_->GetNodeName(), "usedCoreNum: %ld", usedCoreNum_);
    OP_LOGI(context_->GetNodeName(), "dimCoreCnt: %ld", dimCoreCnt_);
    OP_LOGI(context_->GetNodeName(), "batchCoreCnt: %ld", batchCoreCnt_);

    // Dim tiling parameters inter-core
    OP_LOGI(context_->GetNodeName(), "dimChunkSize: %ld", dimChunkSize_);
    OP_LOGI(context_->GetNodeName(), "dimTailSize: %ld", dimTailSize_);

    // Batch tiling parameters inter-core
    OP_LOGI(context_->GetNodeName(), "batchPerCore: %ld", batchPerCore_);
    OP_LOGI(context_->GetNodeName(), "batchTailPerCore: %ld", batchTailPerCore_);
    OP_LOGI(context_->GetNodeName(), "validBatchStart: %ld", validBatchStart_);
    OP_LOGI(context_->GetNodeName(), "validBatchEnd: %ld", validBatchEnd_);

    // Intra-core tiling parameters UB loop
    OP_LOGI(context_->GetNodeName(), "ubBatchSize: %ld", ubBatchSize_);
    OP_LOGI(context_->GetNodeName(), "ubDimSize: %ld", ubDimSize_);
    OP_LOGI(context_->GetNodeName(), "batchLoopCnt: %ld", batchLoopCnt_);
    OP_LOGI(context_->GetNodeName(), "dimLoopCnt: %ld", dimLoopCnt_);

    // Shape information for kernel use
    OP_LOGI(context_->GetNodeName(), "batchSize: %ld", batchSize_);
    OP_LOGI(context_->GetNodeName(), "seqLen: %ld", seqLen_);
    OP_LOGI(context_->GetNodeName(), "cuSeqLen: %ld", cuSeqLen_);
    OP_LOGI(context_->GetNodeName(), "dim: %ld", dim_);
    OP_LOGI(context_->GetNodeName(), "kernelSize: %ld", kernelSize_);
    OP_LOGI(context_->GetNodeName(), "stateLen: %ld", stateLen_);
    OP_LOGI(context_->GetNodeName(), "xInputMode: %ld", xInputMode_);
    OP_LOGI(context_->GetNodeName(), "hasAcceptTokenNum: %ld", hasAcceptTokenNum_);

    // Additional debug information (not in struct)
    OP_LOGI(context_->GetNodeName(), "Invalid Batch Number: %ld", inValidBatchNum_);
    OP_LOGI(context_->GetNodeName(), "Total Core Number: %ld", totalCoreNum_);
    OP_LOGI(context_->GetNodeName(), "Limited Core Number: %ld", limitedCoreNum_);
    OP_LOGI(context_->GetNodeName(), "UB Size: %lu bytes", ubSize_);
}

ge::graphStatus CausalConv1dUpdateTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdateTiling::GetWorkspaceSize()
{
    auto platformInfo = context_->GetPlatformInfo();
    auto ascendcPlatform = platform_ascendc::PlatformAscendC(platformInfo);
    uint32_t sysWorkspaceSize = ascendcPlatform.GetLibApiWorkSpaceSize();
    size_t *currentWorkspace = context_->GetWorkspaceSizes(1);
    currentWorkspace[0] = static_cast<size_t>(0UL + sysWorkspaceSize);
    return ge::GRAPH_SUCCESS;
}

// REGISTER_OPS_TILING_TEMPLATE(CausalConv1dUpdate, CausalConv1dUpdateTiling, 1);

} // namespace optiling
