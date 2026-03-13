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
#define TILING_KEY_UPDATE_FP16 20001

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

    const int64_t* residualConnectionPtr = attrs->GetAttrPointer<int64_t>(ATTR_RESIDUAL_CONNECTION_INDEX);
    if (residualConnectionPtr != nullptr) {
        residualConnection_ = *residualConnectionPtr;
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

        // New 2D tiling (non-uniform): first split dim by 128, then split batch to maximize cores
    const int64_t DIM_GRANULARITY = DIM_ALIGN_ELEMENT; // 128 elements (256B)
    int64_t N = dim_ / DIM_GRANULARITY;
    if (N <= 0) {
        OP_LOGE(context_->GetNodeName(), "dim %ld is smaller than DIM_GRANULARITY %ld", dim_, DIM_GRANULARITY);
        return ge::GRAPH_FAILED;
    }

    // Compute max available cores considering data-size limit
    limitedCoreNum_ = CalculateLimitedCoreNum();
    int64_t maxCoresAvailable = std::min<int64_t>(totalCoreNum_, limitedCoreNum_);
    printf("limitedCoreNum_ = %u", limitedCoreNum_);
    
    // Greedy search best (dimCores, bsCores), prioritize more dim splits
    int64_t bestDimCores = 1;
    int64_t bestBSCores = 1;
    int64_t bestUsed = 1; // dc=1 initially

    for (int64_t dc = N; dc >= 1; --dc) {
        int64_t baseBlocks = N / dc; // each small core gets base blocks of 128

        int64_t maxAllowedBSByCore = maxCoresAvailable / dc;
        if (maxAllowedBSByCore == 0) continue;
        int64_t actualBS = std::min<int64_t>(validBatch, maxAllowedBSByCore);
        if (actualBS <= 0) continue;
        int64_t usedCores = dc * actualBS;
        if (usedCores > bestUsed || (usedCores == bestUsed && dc > bestDimCores)) {
            bestDimCores = dc;
            bestBSCores = actualBS;
            bestUsed = usedCores;
        }
        if (bestUsed == maxCoresAvailable) {
            break; // fully utilized
        }
    }

    // Derive dim non-uniform parameters
    int64_t base = N / bestDimCores;
    int64_t remainder = N % bestDimCores; // big core count

    dimCoreCnt_ = bestDimCores;
    dimMainCoreCnt_ = remainder;
    dimTailCoreCnt_ = bestDimCores - remainder;
    if (remainder > 0) {
        dimChunkSize_ = (base + 1) * DIM_GRANULARITY; // big core size
        dimTailSize_  = base * DIM_GRANULARITY;       // small core size
    } else {
        dimChunkSize_ = base * DIM_GRANULARITY;
        dimTailSize_  = base * DIM_GRANULARITY;
    }

    // Derive batch non-uniform parameters (均分+多前核)
    batchCoreCnt_ = bestBSCores;
    int64_t bsBase = validBatch / batchCoreCnt_;
    int64_t bsRemainder = validBatch % batchCoreCnt_;
    batchMainCoreCnt_ = bsRemainder;               // 前remainder个核是大核
    batchTailCoreCnt_ = batchCoreCnt_ - bsRemainder; // 其余是小核
    batchMainPerCore_ = bsBase + (bsRemainder > 0 ? 1 : 0); // 大核批大小
    batchTailPerCore_ = bsBase;                          // 小核批大小

    usedCoreNum_ = dimCoreCnt_ * batchCoreCnt_;

    // Intra-core UB tiling (big and tail blocks separately), similar to Fn
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - fixedUBSize;
    auto computeUbFor = [&](int64_t coreDimElems, int64_t coreBS, int64_t &outUbDim, int64_t &outUbBS,
                            int64_t &outLoopDim, int64_t &outLoopBS, int64_t &outUbTailDim, int64_t &outUbTailBS) -> void {
        // weight + convStates per-dim element coefficient
        int64_t weightConvStatesCoeffPerDim = (kernelSize_ + kernelSize_ + seqLen_ - 2) * DTYPE_SIZE;
        int64_t xCoeffPerDimFullBS = BUFFER_NUM * coreBS * seqLen_ * DTYPE_SIZE;
        int64_t totalCoeffPerDim = weightConvStatesCoeffPerDim + xCoeffPerDimFullBS;
        int64_t maxUbDim = (totalCoeffPerDim > 0) ? (availableUbSize / totalCoeffPerDim) : 0;
        maxUbDim = (maxUbDim / DIM_GRANULARITY) * DIM_GRANULARITY;
        if (maxUbDim >= DIM_GRANULARITY) {
            outUbBS = coreBS;
            outUbDim = std::min(maxUbDim, coreDimElems);
            outUbDim = (outUbDim / DIM_GRANULARITY) * DIM_GRANULARITY;
            if (outUbDim == 0) outUbDim = DIM_GRANULARITY;
        } else {
            outUbDim = DIM_GRANULARITY;
            int64_t weightConvStatesSize = weightConvStatesCoeffPerDim * outUbDim;
            int64_t availableForX = availableUbSize - weightConvStatesSize;
            int64_t xSizePerBatch = seqLen_ * outUbDim * DTYPE_SIZE;
            outUbBS = (xSizePerBatch > 0) ? (availableForX / xSizePerBatch) : 0;
            outUbBS = std::max<int64_t>(outUbBS, 1);
            outUbBS = std::min(outUbBS, coreBS);
        }
        outLoopDim = (coreDimElems + outUbDim - 1) / outUbDim;
        outUbTailDim = (outLoopDim == 1) ? outUbDim : (coreDimElems - (outLoopDim - 1) * outUbDim);
        outLoopBS = (coreBS + outUbBS - 1) / outUbBS;
        outUbTailBS = (outLoopBS == 1) ? outUbBS : (coreBS - (outLoopBS - 1) * outUbBS);
    };

    // Big cores UB params
    computeUbFor(dimChunkSize_, batchMainPerCore_, ubMainFactorDim_, ubMainFactorBS_, loopNumDim_, loopNumBS_, ubTailFactorDim_, ubTailFactorBS_);
    // Tail cores UB params
    computeUbFor((dimMainCoreCnt_ > 0 ? dimTailSize_ : dimChunkSize_), batchTailPerCore_,
                 tailBlockubFactorDim_, tailBlockubFactorBS_, tailBlockloopNumDim_, tailBlockloopNumBS_,
                 tailBlockubTailFactorDim_, tailBlockubTailFactorBS_);

    return ge::GRAPH_SUCCESS;
}

int64_t CausalConv1dUpdateTiling::CalculateLimitedCoreNum()
{
    // Calculate input x size in bytes (data type is float16/bf16, 2 bytes per element)
    int64_t xSizeBytes;
    if (xInputMode_ == X_INPUT_3D) {
        // 3D input: batchSize * seqLen * dim * 2 bytes
        // xSizeBytes = batchSize_ * seqLen_ * dim_ * DTYPE_SIZE;
        xSizeBytes = batchSize_ * dim_ * DTYPE_SIZE;
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
    // New rule: increase 1 core per additional 256 bytes of x data
    // effectiveCoreNum = ceil(xSizeBytes / 256)
    const int64_t bytesPerCoreStep = 256;
    int64_t effectiveCoreNum = (xSizeBytes + bytesPerCoreStep - 1) / bytesPerCoreStep;
    effectiveCoreNum = std::max(effectiveCoreNum, static_cast<int64_t>(1));

    // Actual core number is min(effectiveCoreNum, totalCoreNum_)
    return std::min(effectiveCoreNum, static_cast<int64_t>(totalCoreNum_));
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

    // Dim tiling parameters (non-uniform)
    tilingData_.dimMainCoreCnt = dimMainCoreCnt_;
    tilingData_.dimTailCoreCnt = dimTailCoreCnt_;
    tilingData_.dimChunkSize = dimChunkSize_;
    tilingData_.dimTailSize = dimTailSize_;

    // Batch tiling parameters (non-uniform)
    tilingData_.batchMainCoreCnt = batchMainCoreCnt_;
    tilingData_.batchTailCoreCnt = batchTailCoreCnt_;
    tilingData_.batchMainPerCore = batchMainPerCore_;
    tilingData_.batchTailPerCore = batchTailPerCore_;
    tilingData_.validBatchStart = validBatchStart_;
    tilingData_.validBatchEnd = validBatchEnd_;

    // Intra-core tiling parameters (UB loop, big/tail blocks)
    tilingData_.loopNumBS = loopNumBS_;
    tilingData_.loopNumDim = loopNumDim_;
    tilingData_.ubMainFactorBS = ubMainFactorBS_;
    tilingData_.ubTailFactorBS = ubTailFactorBS_;
    tilingData_.ubMainFactorDim = ubMainFactorDim_;
    tilingData_.ubTailFactorDim = ubTailFactorDim_;
    tilingData_.tailBlockloopNumBS = tailBlockloopNumBS_;
    tilingData_.tailBlockloopNumDim = tailBlockloopNumDim_;
    tilingData_.tailBlockubFactorBS = tailBlockubFactorBS_;
    tilingData_.tailBlockubTailFactorBS = tailBlockubTailFactorBS_;
    tilingData_.tailBlockubFactorDim = tailBlockubFactorDim_;
    tilingData_.tailBlockubTailFactorDim = tailBlockubTailFactorDim_;

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
    tilingData_.residualConnection = residualConnection_;

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
    OP_LOGI(context_->GetNodeName(), "batchMainPerCore: %ld", batchMainPerCore_);
    OP_LOGI(context_->GetNodeName(), "batchTailPerCore: %ld", batchTailPerCore_);
    OP_LOGI(context_->GetNodeName(), "validBatchStart: %ld", validBatchStart_);
    OP_LOGI(context_->GetNodeName(), "validBatchEnd: %ld", validBatchEnd_);

    // Intra-core tiling parameters UB loop (big/tail blocks)
    OP_LOGI(context_->GetNodeName(), "ubMainFactorBS: %ld", ubMainFactorBS_);
    OP_LOGI(context_->GetNodeName(), "ubMainFactorDim: %ld", ubMainFactorDim_);
    OP_LOGI(context_->GetNodeName(), "loopNumBS: %ld", loopNumBS_);
    OP_LOGI(context_->GetNodeName(), "loopNumDim: %ld", loopNumDim_);
    OP_LOGI(context_->GetNodeName(), "tailBlockubFactorBS: %ld", tailBlockubFactorBS_);
    OP_LOGI(context_->GetNodeName(), "tailBlockubFactorDim: %ld", tailBlockubFactorDim_);
    OP_LOGI(context_->GetNodeName(), "tailBlockloopNumBS: %ld", tailBlockloopNumBS_);
    OP_LOGI(context_->GetNodeName(), "tailBlockloopNumDim: %ld", tailBlockloopNumDim_);

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

REGISTER_OPS_TILING_TEMPLATE(CausalConv1dUpdate, CausalConv1dUpdateTiling, 1);

} // namespace optiling
