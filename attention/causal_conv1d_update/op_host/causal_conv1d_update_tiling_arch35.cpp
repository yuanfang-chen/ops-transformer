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
#include "../op_kernal/arch35/causal_conv1d_update_struct.h"
#include <algorithm>

namespace optiling {

// Constants for validation
constexpr int64_t ALIGN_SIZE = 256;
constexpr int64_t MIN_DIM = 64;
constexpr int64_t MAX_DIM = 16384;
constexpr int64_t MIN_BATCH = 1;
constexpr int64_t MAX_BATCH = 256;
constexpr int64_t MIN_M = 0;
constexpr int64_t MAX_M = 5;

constexpr uint64_t TILING_KEY_BASE = 30000UL;

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
    if (xOriginShape.GetDimNum() == 3) {
        xInputMode_ = 0;  // 3D input mode
        batchSize_ = xOriginShape.GetDim(0);
        seqLen_ = xOriginShape.GetDim(1);
        dim_ = xOriginShape.GetDim(2);
    } else if (xOriginShape.GetDimNum() == 2) {
        xInputMode_ = 1;  // 2D input mode
        cuSeqLen_ = xOriginShape.GetDim(0);  // cu_seq_len = batch * seq_len
        dim_ = xOriginShape.GetDim(1);

        // For 2D input, query_start_loc must be specified to get batch
        auto queryStartLocShape = context_->GetInputShape(QUERY_START_LOC_INDEX);
        OP_CHECK_NULL_WITH_CONTEXT(context_, queryStartLocShape);
        auto queryStartLocOriginShape = queryStartLocShape->GetOriginShape();

        // query_start_loc shape is (batch + 1,), so batch = dim0 - 1
        batchSize_ = queryStartLocOriginShape.GetDim(0) - 1;
    } else {
        OP_LOGE(context_->GetNodeName(), "X dimension number must be 2 or 3, but got %lu",
                xOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    // Get weight shape
    auto weightShape = context_->GetInputShape(WEIGHT_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, weightShape);
    auto weightOriginShape = weightShape->GetOriginShape();

    if (weightOriginShape.GetDimNum() != 2) {
        OP_LOGE(context_->GetNodeName(), "Weight dimension number must be 2, but got %lu",
                weightOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    kernelSize_ = weightOriginShape.GetDim(0);

    // Get data types
    xDtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    weightDtype_ = context_->GetInputDesc(WEIGHT_INDEX)->GetDataType();
    convStatesDtype_ = context_->GetInputDesc(CONV_STATES_INDEX)->GetDataType();
    queryStartLocDtype_ = context_->GetInputDesc(QUERY_START_LOC_INDEX)->GetDataType();
    cacheIndicesDtype_ = context_->GetInputDesc(CACHE_INDICES_INDEX)->GetDataType();

    // Get numAcceptedTokens dtype if available (optional input)
    auto numAcceptedTokensDesc = context_->GetOptionalInputDesc(NUM_ACCEPTED_TOKENS_INDEX);
    if (numAcceptedTokensDesc != nullptr) {
        numAcceptedTokensDtype_ = numAcceptedTokensDesc->GetDataType();
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

    const int64_t* padSlotIdPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_ID_INDEX);
    if (padSlotIdPtr != nullptr) {
        padSlotId_ = *padSlotIdPtr;
    }

    const int64_t* residualConnModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RESIDUAL_CONN_MODE_INDEX);
    if (residualConnModePtr != nullptr) {
        residualConnMode_ = *residualConnModePtr;
    }

    const int64_t* runModePtr = attrs->GetAttrPointer<int64_t>(ATTR_RUN_MODE_INDEX);
    if (runModePtr != nullptr) {
        runMode_ = *runModePtr;
    }

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
    if (xInputMode_ == 0) {
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
    OP_CHECK_IF(convStatesOriginShape.GetDimNum() != 3,
                OP_LOGE(context_->GetNodeName(),
                        "ConvStates dimension number must be 3, but got %lu",
                        convStatesOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // For 3D input, validate conv states shape: [-1, K-1+m, dim]
    // The second dimension should be K-1 + m = K-1 + (seqLen-1) = K + seqLen - 2
    if (xInputMode_ == 0) {
        int64_t expectedCacheLen = kernelSize_ + seqLen_ - 2;
        int64_t cacheLen = convStatesOriginShape.GetDim(1);
        OP_CHECK_IF(cacheLen != expectedCacheLen,
                    OP_LOGE(context_->GetNodeName(),
                            "ConvStates length must be K-1+m = %ld, but got %ld",
                            expectedCacheLen, cacheLen),
                    return ge::GRAPH_FAILED);
    }
    // For 2D input, skip seqLen-based validation as seqLen_ is not set

    // Validate conv states dim matches x dim
    int64_t convStatesDim = convStatesOriginShape.GetDim(2);
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
    auto indicesShape = context_->GetInputShape(CACHE_INDICES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, indicesShape);
    auto indicesOriginShape = indicesShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(indicesOriginShape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(),
                        "CacheIndices dimension number must be 1, but got %lu",
                        indicesOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate shape matches batch size
    int64_t indicesLen = indicesOriginShape.GetDim(0);
    OP_CHECK_IF(indicesLen != batchSize_,
                OP_LOGE(context_->GetNodeName(),
                        "CacheIndices length must match batch size %ld, but got %ld",
                        batchSize_, indicesLen),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate accept token num tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateNumAcceptedTokensShape()
{
    // This is an optional input
    if (context_->GetOptionalInputTensor(NUM_ACCEPTED_TOKENS_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    auto acceptShape = context_->GetInputShape(NUM_ACCEPTED_TOKENS_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, acceptShape);
    auto acceptOriginShape = acceptShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(acceptOriginShape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(),
                        "NumAcceptedTokens dimension number must be 1, but got %lu",
                        acceptOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate shape matches batch size
    int64_t acceptLen = acceptOriginShape.GetDim(0);
    OP_CHECK_IF(acceptLen != batchSize_,
                OP_LOGE(context_->GetNodeName(),
                        "NumAcceptedTokens length must match batch size %ld, but got %ld",
                        batchSize_, acceptLen),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate query start loc tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateQueryStartLocShape()
{
    auto queryStartLocShape = context_->GetInputShape(QUERY_START_LOC_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, queryStartLocShape);
    auto queryStartLocOriginShape = queryStartLocShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(queryStartLocOriginShape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(),
                        "QueryStartLoc dimension number must be 1, but got %lu",
                        queryStartLocOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate shape: should be (batch + 1,)
    int64_t queryStartLocLen = queryStartLocOriginShape.GetDim(0);
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
    OP_CHECK_IF(queryStartLocDtype_ != ge::DataType::DT_INT32,
                OP_LOGE(context_->GetNodeName(),
                        "QueryStartLoc data type must be INT32, but got %s",
                        Ops::Base::ToString(queryStartLocDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate num accepted tokens tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateNumAcceptedTokensType()
{
    // This is an optional input
    if (context_->GetOptionalInputTensor(NUM_ACCEPTED_TOKENS_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(numAcceptedTokensDtype_ != ge::DataType::DT_INT32,
                OP_LOGE(context_->GetNodeName(),
                        "NumAcceptedTokens data type must be INT32, but got %s",
                        Ops::Base::ToString(numAcceptedTokensDtype_).c_str()),
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

    OP_CHECK_IF(ValidateNumAcceptedTokensShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "NumAcceptedTokens shape validation failed"),
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

    OP_CHECK_IF(ValidateNumAcceptedTokensType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "NumAcceptedTokens type validation failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdateTiling::DoOpTiling()
{
    // Calculate invalid batch number by checking padSlotId
    // Invalid batches can be at the beginning or at the end
    int64_t invalidBatchAtStart = 0;
    int64_t invalidBatchAtEnd = 0;

    auto cacheIndicesTensor = context_->GetInputTensor(CACHE_INDICES_INDEX);
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
    if (xInputMode_ == 0) {
        // 3D input: batchSize * seqLen * dim * 2 bytes
        xSizeBytes = batchSize_ * seqLen_ * dim_ * 2;
    } else {
        // 2D input: cuSeqLen * dim * 2 bytes
        xSizeBytes = cuSeqLen_ * dim_ * 2;
    }

    // Limit core number based on data size
    // When data occupies half of UB, bandwidth is good
    // effectiveCoreNum = xSizeBytes / (ubSize_ / 2)
    int64_t halfUbSize = ubSize_ / 2;
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

    constexpr int64_t DIM_ALIGN_ELEMENT = 128;  // 256 bytes / 2 bytes per element
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
    constexpr int64_t DIM_ALIGN_ELEMENT = 128;  // 256 bytes / 2 bytes per element

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
    // Only support 3D input for now
    if (xInputMode_ != 0) {
        // For 2D input, set default values
        ubBatchSize_ = batchPerCore_;
        ubDimSize_ = dimChunkSize_;
        batchLoopCnt_ = 1;
        dimLoopCnt_ = 1;
        return;
    }

    // Fixed UB usage for auxiliary tensors
    // queryStartLoc: 257 * sizeof(int32) = 1028 bytes
    // cacheIndices: 256 * sizeof(int32) = 1024 bytes
    // numAcceptedTokens: 257 * sizeof(int32) = 1028 bytes
    constexpr int64_t QUERY_START_LOC_UB_SIZE = 257 * sizeof(int32_t);
    constexpr int64_t CACHE_INDICES_UB_SIZE = 256 * sizeof(int32_t);
    constexpr int64_t NUM_ACCEPTED_TOKENS_UB_SIZE = 257 * sizeof(int32_t);
    constexpr int64_t FIXED_UB_SIZE = QUERY_START_LOC_UB_SIZE + CACHE_INDICES_UB_SIZE + NUM_ACCEPTED_TOKENS_UB_SIZE;

    constexpr int64_t DIM_ALIGN_ELEMENTS = 128;  // 256 bytes / 2 bytes per bf16
    constexpr int64_t DTYPE_SIZE = 2;  // bf16/fp16 size in bytes

    // Use dimChunkSize_ and batchPerCore_ as the maximum data per core
    int64_t coreDim = dimChunkSize_;
    int64_t coreBatch = batchPerCore_;

    // Coefficient per dim element for weight and convStates (fixed per UB iteration)
    // weight: kernelSize_ * ubDim * DTYPE_SIZE
    // convStates: (kernelSize_ + m - 1) * ubDim * DTYPE_SIZE, where m = seqLen_ - 1
    //           = (kernelSize_ + seqLen_ - 2) * ubDim * DTYPE_SIZE
    int64_t weightConvStatesCoeffPerDim = (kernelSize_ + kernelSize_ + seqLen_ - 2) * DTYPE_SIZE;

    // x coefficient per dim element when full batch is loaded
    // x: coreBatch * seqLen_ * ubDim * DTYPE_SIZE
    int64_t xCoeffPerDimFullBatch = coreBatch * seqLen_ * DTYPE_SIZE;

    // Total coefficient per dim element with full batch
    int64_t totalCoeffPerDim = weightConvStatesCoeffPerDim + xCoeffPerDimFullBatch;

    // Calculate maximum ubDim when full batch is loaded
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - FIXED_UB_SIZE;
    int64_t maxUbDim = availableUbSize / totalCoeffPerDim;

    // Align to DIM_ALIGN_ELEMENTS (256 bytes = 128 bf16 elements)
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;

    if (maxUbDim >= DIM_ALIGN_ELEMENTS) {
        // Can load full batch, try to maximize dim
        ubBatchSize_ = coreBatch;
        ubDimSize_ = std::min(maxUbDim, coreDim);

        // Ensure ubDimSize_ is aligned to DIM_ALIGN_ELEMENTS
        ubDimSize_ = (ubDimSize_ / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;
        if (ubDimSize_ == 0) {
            ubDimSize_ = DIM_ALIGN_ELEMENTS;
        }
    } else {
        // Cannot load full batch with minimum dim, need to reduce batch
        // Use minimum ubDim = DIM_ALIGN_ELEMENTS
        ubDimSize_ = DIM_ALIGN_ELEMENTS;

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
    return TILING_KEY_BASE;
}

ge::graphStatus CausalConv1dUpdateTiling::PostTiling()
{
    // Set block dimension (number of cores to use)
    context_->SetBlockDim(usedCoreNum_);

    // Populate tiling data - core distribution
    tilingData_.set_usedCoreNum(usedCoreNum_);
    tilingData_.set_dimCoreCnt(dimCoreCnt_);
    tilingData_.set_batchCoreCnt(batchCoreCnt_);

    // Dim tiling parameters
    tilingData_.set_dimChunkSize(dimChunkSize_);
    tilingData_.set_dimTailSize(dimTailSize_);

    // Batch tiling parameters
    tilingData_.set_batchPerCore(batchPerCore_);
    tilingData_.set_batchTailPerCore(batchTailPerCore_);
    tilingData_.set_validBatchStart(validBatchStart_);
    tilingData_.set_validBatchEnd(validBatchEnd_);

    // Intra-core tiling parameters (UB loop)
    tilingData_.set_ubBatchSize(ubBatchSize_);
    tilingData_.set_ubDimSize(ubDimSize_);
    tilingData_.set_batchLoopCnt(batchLoopCnt_);
    tilingData_.set_dimLoopCnt(dimLoopCnt_);

    // Shape information for kernel use
    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_seqLen(seqLen_);
    tilingData_.set_cuSeqLen(cuSeqLen_);
    tilingData_.set_dim(dim_);
    tilingData_.set_kernelSize(kernelSize_);
    tilingData_.set_xInputMode(xInputMode_);

    // Save tiling data to buffer
    tilingData_.SaveToBuffer(context_->GetRawTilingData()->GetData(),
                             context_->GetRawTilingData()->GetCapacity());
    context_->GetRawTilingData()->SetDataSize(tilingData_.GetDataSize());

    return ge::GRAPH_SUCCESS;
}

void CausalConv1dUpdateTiling::DumpTilingInfo()
{
    std::ostringstream info;

    // Input shape information
    info << "=== CausalConv1dUpdate Tiling Info ===" << std::endl;
    info << "X Input Mode: " << (xInputMode_ == 0 ? "3D [batch, seq_len, dim]" : "2D [cu_seq_len, dim]") << std::endl;
    info << "Batch Size: " << batchSize_ << std::endl;
    if (xInputMode_ == 0) {
        info << "Sequence Length: " << seqLen_ << std::endl;
    } else {
        info << "CuSeqLen: " << cuSeqLen_ << std::endl;
    }
    info << "Dimension: " << dim_ << std::endl;
    info << "Kernel Size: " << kernelSize_ << std::endl;
    info << "Invalid Batch Number: " << inValidBatchNum_ << std::endl;

    // Hardware information
    info << "Total Core Number: " << totalCoreNum_ << std::endl;
    info << "Limited Core Number: " << limitedCoreNum_ << std::endl;
    info << "Used Core Number: " << usedCoreNum_ << std::endl;
    info << "UB Size: " << ubSize_ << " bytes" << std::endl;

    // Core distribution
    info << "Dim Core Count: " << dimCoreCnt_ << std::endl;
    info << "Batch Core Count: " << batchCoreCnt_ << std::endl;

    // Dim tiling parameters
    info << "Dim Chunk Size: " << dimChunkSize_ << std::endl;
    info << "Dim Tail Size: " << dimTailSize_ << std::endl;

    // Batch tiling parameters
    info << "Batch Per Core: " << batchPerCore_ << std::endl;
    info << "Batch Tail Per Core: " << batchTailPerCore_ << std::endl;
    info << "Valid Batch Start: " << validBatchStart_ << std::endl;
    info << "Valid Batch End: " << validBatchEnd_ << std::endl;

    // Intra-core tiling parameters
    info << "UB Batch Size: " << ubBatchSize_ << std::endl;
    info << "UB Dim Size: " << ubDimSize_ << std::endl;
    info << "Batch Loop Count: " << batchLoopCnt_ << std::endl;
    info << "Dim Loop Count: " << dimLoopCnt_ << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

ge::graphStatus CausalConv1dUpdateTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdateTiling::GetWorkspaceSize()
{
    return ge::GRAPH_SUCCESS;
}

REGISTER_OPS_TILING_TEMPLATE(CausalConv1dUpdate, CausalConv1dUpdateTiling, 1);

} // namespace optiling
