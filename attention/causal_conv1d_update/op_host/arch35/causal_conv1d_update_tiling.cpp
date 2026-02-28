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
 * \file causal_conv1d_update_tiling.cpp
 * \brief CausalConv1dUpdate tiling implementation
 */

#include "causal_conv1d_update_tiling.h"
#include <algorithm>

namespace optiling {

// Constants for validation
constexpr int64_t ALIGN_SIZE = 256;
constexpr int64_t MIN_DIM = 64;
constexpr int64_t MAX_DIM = 16384;
constexpr int64_t MIN_BATCH = 1;
constexpr int64_t MAX_BATCH = 256;
constexpr int64_t EXPECTED_KERNEL_SIZE = 3;
constexpr int64_t MIN_M = 0;
constexpr int64_t MAX_M = 5;
constexpr int64_t DOUBLE_BUFFER_NUM = 2;

constexpr uint64_t TILING_KEY_BASE = 30000UL;

bool CausalConv1dUpdateTiling::IsCapable()
{
    return true;
}

ge::graphStatus AttentionUpdateTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        auto compileInfoPtr = reinterpret_cast<const DecodeUpdateCompileInfo *>(context_->GetCompileInfo());
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

    if (xOriginShape.GetDimNum() != 3) {
        OP_LOGE(context_->GetNodeName(), "X dimension number must be 3, but got %lu",
                xOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    batchSize_ = xOriginShape.GetDim(0);
    seqLen_ = xOriginShape.GetDim(1);
    dim_ = xOriginShape.GetDim(2);

    // Get filter shape
    auto filterShape = context_->GetInputShape(FILTER_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, filterShape);
    auto filterOriginShape = filterShape->GetOriginShape();

    if (filterOriginShape.GetDimNum() != 2) {
        OP_LOGE(context_->GetNodeName(), "Filter dimension number must be 2, but got %lu",
                filterOriginShape.GetDimNum());
        return ge::GRAPH_FAILED;
    }

    kernelSize_ = filterOriginShape.GetDim(0);

    // Get data types
    xDtype_ = context_->GetInputDesc(X_INDEX)->GetDataType();
    filterDtype_ = context_->GetInputDesc(FILTER_INDEX)->GetDataType();
    cacheStateDtype_ = context_->GetInputDesc(CACHE_STATE_INDEX)->GetDataType();
    cacheIndicesDtype_ = context_->GetInputDesc(CACHE_INDICES_INDEX)->GetDataType();

    // Get acceptTokenNum dtype if available
    if (context_->GetInputTensor(ACCEPT_TOKEN_NUM_INDEX) != nullptr) {
        acceptTokenNumDtype_ = context_->GetInputDesc(ACCEPT_TOKEN_NUM_INDEX)->GetDataType();
    }

    // Get dtype size
    xDtypeSize_ = GetSizeByDataType(xDtype_);
    OP_CHECK_IF(xDtypeSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dUpdate get x dtype[%s] size is 0.",
                        Ops::Base::ToString(xDtype_).c_str()),
                return ge::GRAPH_FAILED);

    // Get padSlotIndex attribute
    auto attrs = context_->GetAttrs();
    OP_CHECK_NULL_WITH_CONTEXT(context_, attrs);

    const int64_t* padSlotIndexPtr = attrs->GetAttrPointer<int64_t>(ATTR_PAD_SLOT_INDEX_INDEX);
    if (padSlotIndexPtr != nullptr) {
        padSlotIndex_ = *padSlotIndexPtr;
    } else {
        padSlotIndex_ = 0;
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

    // Validate sequence length: m+1 where m in [0, 5], so seqLen in [1, 6]
    int64_t m = seqLen_ - 1;
    OP_CHECK_IF(m < MIN_M || m > MAX_M,
                OP_LOGE(context_->GetNodeName(),
                        "X sequence length must be m+1 where m in [%ld, %ld], but got seqLen=%ld (m=%ld)",
                        MIN_M, MAX_M, seqLen_, m),
                return ge::GRAPH_FAILED);

    // Validate dimension: [64, 16384]
    OP_CHECK_IF(dim_ < MIN_DIM || dim_ > MAX_DIM,
                OP_LOGE(context_->GetNodeName(),
                        "X dimension must be in [%ld, %ld], but got %ld",
                        MIN_DIM, MAX_DIM, dim_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate filter tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateFilterShape()
{
    auto filterShape = context_->GetInputShape(FILTER_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, filterShape);
    auto filterOriginShape = filterShape->GetOriginShape();

    // Validate kernel size: must be 3
    OP_CHECK_IF(kernelSize_ != EXPECTED_KERNEL_SIZE,
                OP_LOGE(context_->GetNodeName(),
                        "Filter kernel size must be %ld, but got %ld",
                        EXPECTED_KERNEL_SIZE, kernelSize_),
                return ge::GRAPH_FAILED);

    // Validate filter dim matches x dim
    int64_t filterDim = filterOriginShape.GetDim(1);
    OP_CHECK_IF(filterDim != dim_,
                OP_LOGE(context_->GetNodeName(),
                        "Filter dimension must match X dimension %ld, but got %ld",
                        dim_, filterDim),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate cache state tensor shape
ge::graphStatus CausalConv1dUpdateTiling::ValidateCacheStateShape()
{
    auto cacheStateShape = context_->GetInputShape(CACHE_STATE_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, cacheStateShape);
    auto cacheStateOriginShape = cacheStateShape->GetOriginShape();

    // Validate dimension number
    OP_CHECK_IF(cacheStateOriginShape.GetDimNum() != 3,
                OP_LOGE(context_->GetNodeName(),
                        "CacheState dimension number must be 3, but got %lu",
                        cacheStateOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate cache state shape: [-1, K-1+m, dim]
    // The second dimension should be K-1 + m = K-1 + (seqLen-1) = K + seqLen - 2
    int64_t expectedCacheLen = kernelSize_ + seqLen_ - 2;
    int64_t cacheLen = cacheStateOriginShape.GetDim(1);
    OP_CHECK_IF(cacheLen != expectedCacheLen,
                OP_LOGE(context_->GetNodeName(),
                        "CacheState length must be K-1+m = %ld, but got %ld",
                        expectedCacheLen, cacheLen),
                return ge::GRAPH_FAILED);

    // Validate cache state dim matches x dim
    int64_t cacheDim = cacheStateOriginShape.GetDim(2);
    OP_CHECK_IF(cacheDim != dim_,
                OP_LOGE(context_->GetNodeName(),
                        "CacheState dimension must match X dimension %ld, but got %ld",
                        dim_, cacheDim),
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
ge::graphStatus CausalConv1dUpdateTiling::ValidateAcceptTokenNumShape()
{
    // This is an optional input
    if (context_->GetInputTensor(ACCEPT_TOKEN_NUM_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    auto acceptShape = context_->GetInputShape(ACCEPT_TOKEN_NUM_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, acceptShape);
    auto acceptOriginShape = acceptShape->GetOriginShape();

    // Validate dimension number: must be 1
    OP_CHECK_IF(acceptOriginShape.GetDimNum() != 1,
                OP_LOGE(context_->GetNodeName(),
                        "AcceptTokenNum dimension number must be 1, but got %lu",
                        acceptOriginShape.GetDimNum()),
                return ge::GRAPH_FAILED);

    // Validate shape matches batch size
    int64_t acceptLen = acceptOriginShape.GetDim(0);
    OP_CHECK_IF(acceptLen != batchSize_,
                OP_LOGE(context_->GetNodeName(),
                        "AcceptTokenNum length must match batch size %ld, but got %ld",
                        batchSize_, acceptLen),
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

// Validate filter tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateFilterType()
{
    OP_CHECK_IF(filterDtype_ != xDtype_,
                OP_LOGE(context_->GetNodeName(),
                        "Filter data type must match X data type %s, but got %s",
                        Ops::Base::ToString(xDtype_).c_str(),
                        Ops::Base::ToString(filterDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate cache state tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateCacheStateType()
{
    OP_CHECK_IF(cacheStateDtype_ != xDtype_,
                OP_LOGE(context_->GetNodeName(),
                        "CacheState data type must match X data type %s, but got %s",
                        Ops::Base::ToString(xDtype_).c_str(),
                        Ops::Base::ToString(cacheStateDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate cache indices tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateCacheIndicesType()
{
    OP_CHECK_IF(cacheIndicesDtype_ != ge::DataType::DT_INT64,
                OP_LOGE(context_->GetNodeName(),
                        "CacheIndices data type must be INT64, but got %s",
                        Ops::Base::ToString(cacheIndicesDtype_).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// Validate accept token num tensor type
ge::graphStatus CausalConv1dUpdateTiling::ValidateAcceptTokenNumType()
{
    // This is an optional input
    if (context_->GetInputTensor(ACCEPT_TOKEN_NUM_INDEX) == nullptr) {
        return ge::GRAPH_SUCCESS;
    }

    OP_CHECK_IF(acceptTokenNumDtype_ != ge::DataType::DT_INT64,
                OP_LOGE(context_->GetNodeName(),
                        "AcceptTokenNum data type must be INT64, but got %s",
                        Ops::Base::ToString(acceptTokenNumDtype_).c_str()),
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

    OP_CHECK_IF(ValidateFilterShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Filter shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateCacheStateShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CacheState shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateCacheIndicesShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CacheIndices shape validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateAcceptTokenNumShape() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AcceptTokenNum shape validation failed"),
                return ge::GRAPH_FAILED);

    // Validate all types
    OP_CHECK_IF(ValidateXType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "X type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateFilterType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "Filter type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateCacheStateType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CacheState type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateCacheIndicesType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CacheIndices type validation failed"),
                return ge::GRAPH_FAILED);

    OP_CHECK_IF(ValidateAcceptTokenNumType() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "AcceptTokenNum type validation failed"),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dUpdateTiling::DoOpTiling()
{
    // Calculate invalid batch number by checking padSlotIndex
    inValidBatchNum_ = 0;
    auto convStateIndicesTensor = context_->GetInputTensor(CACHE_INDICES_INDEX);
    if (convStateIndicesTensor != nullptr) {
        const int64_t* dataPtr = convStateIndicesTensor->GetData<int64_t>();
        if (dataPtr != nullptr) {
            for (int64_t i = batchSize_ - 1; i >= 0; i--) {
                if (padSlotIndex_ == dataPtr[i]) {
                    inValidBatchNum_++;
                } else {
                    break;
                }
            }
        }
    }

    // Calculate valid batch
    int64_t validBatch = batchSize_ - inValidBatchNum_;
    OP_CHECK_IF(validBatch <= 0,
                OP_LOGE(context_->GetNodeName(), "Valid batch must be positive, but got %ld", validBatch),
                return ge::GRAPH_FAILED);

    // Calculate core distribution parameters
    CalculateCoreParams(validBatch);

    // Calculate loop parameters for regular blocks
    CalculateLoopParams(false);

    // Calculate loop parameters for tail block
    CalculateLoopParams(true);

    return ge::GRAPH_SUCCESS;
}

void CausalConv1dUpdateTiling::CalculateCoreParams(int64_t validBatch)
{
    // Calculate block factor and used core number
    blockFactor_ = (validBatch + totalCoreNum_ - 1) / totalCoreNum_;
    usedCoreNum_ = (validBatch + blockFactor_ - 1) / blockFactor_;
    blockTailFactor_ = validBatch - (usedCoreNum_ - 1) * blockFactor_;

    usedCoreNum_ = std::max(usedCoreNum_, static_cast<int64_t>(1));
}

void CausalConv1dUpdateTiling::CalculateLoopParams(bool isTailBlock)
{
    int64_t currentBlockFactor = isTailBlock ? blockTailFactor_ : blockFactor_;

    // Calculate fixed buffer sizes
    int64_t filterBufferSize = kernelSize_ * ALIGN_SIZE;
    int64_t cacheLen = kernelSize_ + seqLen_ - 2;
    int64_t cacheBufferSize = cacheLen * ALIGN_SIZE;
    int64_t indicesBufferSize = ALIGN_SIZE * sizeof(int64_t);
    int64_t acceptTokenBufferSize = ALIGN_SIZE * sizeof(int64_t);

    // Total fixed buffer size
    int64_t fixedBufferSize = filterBufferSize + cacheBufferSize +
                              indicesBufferSize + acceptTokenBufferSize;

    // Remaining UB size for xQueue (with double buffering)
    int64_t remainingUBSize = ubSize_ - fixedBufferSize;
    int64_t xQueueSizePerBuffer = remainingUBSize / DOUBLE_BUFFER_NUM;

    // Calculate AlignElement (256B alignment for dim)
    int64_t AlignElement = ALIGN_SIZE / xDtypeSize_;

    // Calculate BS (batch * sequence) for current block
    int64_t totalBS = currentBlockFactor * seqLen_;

    // Calculate bytes per BS item
    int64_t bytesPerBSItem = AlignElement * xDtypeSize_;

    // Try to maximize n (number of BS items per loop) while keeping dim at AlignElement
    int64_t maxBSPerLoop = xQueueSizePerBuffer / bytesPerBSItem;

    int64_t ubFactorBS;
    int64_t loopNumBS;
    int64_t ubTailFactorBS;
    int64_t ubFactorDim;
    int64_t loopNumDim;
    int64_t ubTailFactorDim;

    if (maxBSPerLoop >= totalBS) {
        // Can fit all BS in one loop
        ubFactorBS = totalBS;
        loopNumBS = 1;
        ubTailFactorBS = totalBS;

        // Try to expand dim if there's remaining space
        int64_t remainingSpace = xQueueSizePerBuffer - totalBS * bytesPerBSItem;
        int64_t additionalElements = remainingSpace / (totalBS * xDtypeSize_);
        // Align to 256B (AlignElement elements)
        additionalElements = (additionalElements / AlignElement) * AlignElement;
        int64_t expandedDim = AlignElement + additionalElements;
        if (expandedDim > dim_) {
            expandedDim = dim_;
        }

        ubFactorDim = expandedDim;
        loopNumDim = (dim_ + ubFactorDim - 1) / ubFactorDim;
        ubTailFactorDim = dim_ - (loopNumDim - 1) * ubFactorDim;
    } else {
        // Need multiple loops for BS dimension
        ubFactorBS = maxBSPerLoop;
        loopNumBS = (totalBS + ubFactorBS - 1) / ubFactorBS;
        ubTailFactorBS = totalBS - (loopNumBS - 1) * ubFactorBS;

        // Dim uses AlignElement size per loop
        ubFactorDim = AlignElement;
        loopNumDim = (dim_ + ubFactorDim - 1) / ubFactorDim;
        ubTailFactorDim = dim_ - (loopNumDim - 1) * ubFactorDim;
    }

    // Save parameters
    if (isTailBlock) {
        tailBlockloopNumBS_ = loopNumBS;
        tailBlockubFactorBS_ = ubFactorBS;
        tailBlockubTailFactorBS_ = ubTailFactorBS;
        tailBlockloopNumDim_ = loopNumDim;
        tailBlockubFactorDim_ = ubFactorDim;
        tailBlockubTailFactorDim_ = ubTailFactorDim;
    } else {
        loopNumBS_ = loopNumBS;
        ubFactorBS_ = ubFactorBS;
        ubTailFactorBS_ = ubTailFactorBS;
        loopNumDim_ = loopNumDim;
        ubFactorDim_ = ubFactorDim;
        ubTailFactorDim_ = ubTailFactorDim;
    }
}

uint64_t CausalConv1dUpdateTiling::GetTilingKey() const
{
    return TILING_KEY_BASE;
}

ge::graphStatus CausalConv1dUpdateTiling::PostTiling()
{
    // Set block dimension (number of cores to use)
    context_->SetBlockDim(usedCoreNum_);

    // Populate tiling data
    tilingData_.set_blockFactor(blockFactor_);
    tilingData_.set_blockTailFactor(blockTailFactor_);

    tilingData_.set_loopNumBS(loopNumBS_);
    tilingData_.set_loopNumDim(loopNumDim_);
    tilingData_.set_ubFactorBS(ubFactorBS_);
    tilingData_.set_ubTailFactorBS(ubTailFactorBS_);
    tilingData_.set_ubFactorDim(ubFactorDim_);
    tilingData_.set_ubTailFactorDim(ubTailFactorDim_);

    tilingData_.set_tailBlockloopNumBS(tailBlockloopNumBS_);
    tilingData_.set_tailBlockloopNumDim(tailBlockloopNumDim_);
    tilingData_.set_tailBlockubFactorBS(tailBlockubFactorBS_);
    tilingData_.set_tailBlockubTailFactorBS(tailBlockubTailFactorBS_);
    tilingData_.set_tailBlockubFactorDim(tailBlockubFactorDim_);
    tilingData_.set_tailBlockubTailFactorDim(tailBlockubTailFactorDim_);

    // Additional shape information for kernel use
    tilingData_.set_batchSize(batchSize_);
    tilingData_.set_seqLen(seqLen_);
    tilingData_.set_dim(dim_);
    tilingData_.set_kernelSize(kernelSize_);

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
    info << "Batch Size: " << batchSize_ << std::endl;
    info << "Sequence Length: " << seqLen_ << std::endl;
    info << "Dimension: " << dim_ << std::endl;
    info << "Kernel Size: " << kernelSize_ << std::endl;
    info << "Invalid Batch Number: " << inValidBatchNum_ << std::endl;

    // Hardware information
    info << "Total Core Number: " << totalCoreNum_ << std::endl;
    info << "Used Core Number: " << usedCoreNum_ << std::endl;
    info << "UB Size: " << ubSize_ << " bytes" << std::endl;
    info << "UB Block Size: " << ubBlockSize_ << " bytes" << std::endl;

    // Core distribution
    info << "Block Factor: " << blockFactor_ << std::endl;
    info << "Block Tail Factor: " << blockTailFactor_ << std::endl;

    // Regular block loop parameters
    info << "Loop Number BS (regular): " << loopNumBS_ << std::endl;
    info << "Loop Number Dim (regular): " << loopNumDim_ << std::endl;
    info << "UB Factor BS (regular): " << ubFactorBS_ << std::endl;
    info << "UB Tail Factor BS (regular): " << ubTailFactorBS_ << std::endl;
    info << "UB Factor Dim (regular): " << ubFactorDim_ << std::endl;
    info << "UB Tail Factor Dim (regular): " << ubTailFactorDim_ << std::endl;

    // Tail block loop parameters
    info << "Tail Block Loop Number BS: " << tailBlockloopNumBS_ << std::endl;
    info << "Tail Block Loop Number Dim: " << tailBlockloopNumDim_ << std::endl;
    info << "Tail Block UB Factor BS: " << tailBlockubFactorBS_ << std::endl;
    info << "Tail Block UB Tail Factor BS: " << tailBlockubTailFactorBS_ << std::endl;
    info << "Tail Block UB Factor Dim: " << tailBlockubFactorDim_ << std::endl;
    info << "Tail Block UB Tail Factor Dim: " << tailBlockubTailFactorDim_ << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(CausalConv1dUpdate, CausalConv1dUpdateTiling, 1);

} // namespace optiling
