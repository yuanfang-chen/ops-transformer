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
 * \file causal_conv1d_fn_tiling_arch35.cpp
 * \brief
 */

#include "causal_conv1d_fn_tiling_arch35.h"

namespace optiling {
constexpr uint64_t DIM_0 = 0;
constexpr uint64_t DIM_1 = 1;
constexpr uint64_t DIM_2 = 2;

constexpr uint64_t INPUT_X_INDEX = 0;
constexpr uint64_t INPUT_WEIGHT_INDEX = 1;
constexpr uint64_t INPUT_CACHE_STATES_INDEX = 2;
constexpr uint64_t INPUT_QUERY_START_LOC_INDEX = 3;
constexpr uint64_t INPUT_CACHE_INDICES_INDEX = 4;
constexpr uint64_t INPUT_INITIAL_STATE_MODE_INDEX = 5;

constexpr int32_t ATTR_ACTIVATION_MODE_INDEX = 0;
constexpr int32_t ATTR_PAD_SLOT_ID_INDEX = 1;
constexpr int32_t ATTR_RUN_MODE_INDEX = 2;
constexpr int32_t ATTR_RESIDUAL_CONNECTION_INDEX = 3;

constexpr uint64_t OUTPUT_Y_INDEX = 0;
constexpr uint64_t OUTPUT_CACHE_STATES_INDEX = 1;

constexpr uint64_t X_DIM_NUM = 2;
constexpr uint64_t WEIGHT_DIM_NUM = 2;
constexpr uint64_t CACHE_STATES_DIM_NUM = 3;
constexpr uint64_t SEQ_START_INDEX_DIM_NUM = 1;

constexpr uint64_t DIM_MIN = 64;
constexpr uint64_t DIM_MAX = 16384;
constexpr uint64_t DIM_ALIGN = 16;
constexpr uint64_t CU_SEQ_LEN_MIN = 1;
constexpr uint64_t CU_SEQ_LEN_MAX = 65536;
constexpr uint64_t BATCH_MIN = 1;
constexpr uint64_t BATCH_MAX = 256;
constexpr uint64_t KERNEL_WIDTH_MAX = 6;

constexpr uint64_t DIM_ALIGN_ELEMENTS = 128;  // 256 bytes / 2 bytes per element (fp16/bf16)
constexpr uint64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024;  // 8 KB system reserved UB space
constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t TILING_KEY_FN_BF16 = 10000UL;
constexpr uint64_t TILING_KEY_FN_FP16 = 10001UL;
constexpr uint64_t SYS_WORKSPACE_SIZE = static_cast<uint64_t>(16 * 1024 * 1024);

bool CausalConv1dFnTiling::IsCapable()
{
    return true;
}

ge::graphStatus CausalConv1dFnTiling::DoLibApiTiling()
{
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::GetPlatformInfo()
{
    ubBlockSize_ = Ops::Base::GetUbBlockSize(context_);
    auto platformInfo = context_->GetPlatformInfo();
    if (platformInfo == nullptr) {
        OP_LOGE(context_->GetNodeName(), "platform info is null");
        return ge::GRAPH_FAILED;
    }

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

    // 核内 UB 有固定 8KB 的系统保留空间，需要扣除
    if (ubSize <= SYSTEM_RESERVED_UB_SIZE) {
        OP_LOGE(context_->GetNodeName(), "ubSize %lu is too small, must be > %lu", ubSize, SYSTEM_RESERVED_UB_SIZE);
        return ge::GRAPH_FAILED;
    }
    ubSize_ = ubSize - SYSTEM_RESERVED_UB_SIZE;

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据类型
ge::graphStatus CausalConv1dFnTiling::CheckInputDtype()
{
    if (xType_ != ge::DataType::DT_FLOAT16 && xType_ != ge::DataType::DT_BF16) {
        OP_LOGE(context_->GetNodeName(), "X dtype must be fp16 or bf16, but got: %s",
                Ops::Base::ToString(xType_).c_str());
        return ge::GRAPH_FAILED;
    }

    if (weightType_ != xType_) {
        OP_LOGE(context_->GetNodeName(), "Weight dtype must equal to X dtype. X dtype: %s, Weight dtype: %s",
                Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(weightType_).c_str());
        return ge::GRAPH_FAILED;
    }

    auto cacheStatesType = context_->GetInputDesc(INPUT_CACHE_STATES_INDEX)->GetDataType();
    if (cacheStatesType != xType_) {
        OP_LOGE(context_->GetNodeName(), "CacheStates dtype must equal to X dtype. X dtype: %s, CacheStates dtype: %s",
                Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(cacheStatesType).c_str());
        return ge::GRAPH_FAILED;
    }

    // 检查 cacheIndices (OPTIONAL)
    auto cacheIndicesDesc = context_->GetOptionalInputDesc(INPUT_CACHE_INDICES_INDEX);
    if (cacheIndicesDesc != nullptr) {
        auto cacheIndicesType = cacheIndicesDesc->GetDataType();
        if (cacheIndicesType != ge::DataType::DT_INT32) {
            OP_LOGE(context_->GetNodeName(), "CacheIndices dtype must be INT32, but got: %s",
                    Ops::Base::ToString(cacheIndicesType).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    // 检查 queryStartLoc (OPTIONAL)
    auto seqStartIndexDesc = context_->GetOptionalInputDesc(INPUT_QUERY_START_LOC_INDEX);
    if (seqStartIndexDesc != nullptr) {
        auto seqStartIndexType = seqStartIndexDesc->GetDataType();
        if (seqStartIndexType != ge::DataType::DT_INT32) {
            OP_LOGE(context_->GetNodeName(), "SeqStartIndex dtype must be INT32, but got: %s",
                    Ops::Base::ToString(seqStartIndexType).c_str());
            return ge::GRAPH_FAILED;
        }
    }

    return ge::GRAPH_SUCCESS;
}

// 检查输入数据维度
ge::graphStatus CausalConv1dFnTiling::CheckInputDim()
{
    // 检查X的维度
    uint64_t xDimNum = xShape_.GetDimNum();
    OP_CHECK_IF(xDimNum != X_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "X dim must be 2, but got: %lu", xDimNum),
                return ge::GRAPH_FAILED);

    // 检查cu_seq_len范围
    OP_CHECK_IF(!(cuSeqLen_ >= CU_SEQ_LEN_MIN && cuSeqLen_ <= CU_SEQ_LEN_MAX),
                OP_LOGE(context_->GetNodeName(), "cu_seq_len must in [%lu, %lu], but got: %lu",
                        CU_SEQ_LEN_MIN, CU_SEQ_LEN_MAX, cuSeqLen_),
                return ge::GRAPH_FAILED);

    // 检查dim范围和对齐
    OP_CHECK_IF(!(dim_ >= DIM_MIN && dim_ <= DIM_MAX && dim_ % DIM_ALIGN == 0),
                OP_LOGE(context_->GetNodeName(), "dim must in [%lu, %lu] and be multiple of %lu, but got: %lu",
                        DIM_MIN, DIM_MAX, DIM_ALIGN, dim_),
                return ge::GRAPH_FAILED);

    // 检查weight的维度
    uint64_t weightDimNum = weightShape_.GetDimNum();
    OP_CHECK_IF(weightDimNum != WEIGHT_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "Weight dim must be 2, but got: %lu", weightDimNum),
                return ge::GRAPH_FAILED);

    uint64_t weightDim = weightShape_.GetDim(DIM_1);
    OP_CHECK_IF(weightDim != dim_,
                OP_LOGE(context_->GetNodeName(), "Weight dim[1] must equal to X dim[1], X dim: %lu, Weight dim: %lu",
                        dim_, weightDim),
                return ge::GRAPH_FAILED);

    // 检查kernel width
    OP_CHECK_IF(kernelWidth_ > KERNEL_WIDTH_MAX,
                OP_LOGE(context_->GetNodeName(), "Kernel width must <= %lu, but got: %u",
                        KERNEL_WIDTH_MAX, kernelWidth_),
                return ge::GRAPH_FAILED);

    // 检查cacheStates的维度
    uint64_t cacheStatesDimNum = cacheStatesShape_.GetDimNum();
    OP_CHECK_IF(cacheStatesDimNum != CACHE_STATES_DIM_NUM,
                OP_LOGE(context_->GetNodeName(), "CacheStates dim must be 3, but got: %lu", cacheStatesDimNum),
                return ge::GRAPH_FAILED);

    uint64_t cacheStatesDim1 = cacheStatesShape_.GetDim(DIM_1);
    OP_CHECK_IF(cacheStatesDim1 != (kernelWidth_ - 1),
                OP_LOGE(context_->GetNodeName(), "CacheStates dim[1] must equal to K-1=%u, but got: %lu",
                        kernelWidth_ - 1, cacheStatesDim1),
                return ge::GRAPH_FAILED);

    uint64_t cacheStatesDim2 = cacheStatesShape_.GetDim(DIM_2);
    OP_CHECK_IF(cacheStatesDim2 != dim_,
                OP_LOGE(context_->GetNodeName(), "CacheStates dim[2] must equal to dim=%lu, but got: %lu",
                        dim_, cacheStatesDim2),
                return ge::GRAPH_FAILED);

    // 检查seqStartIndex的维度 (必须提供)
    auto seqStartIndexStorageShape = context_->GetOptionalInputShape(INPUT_QUERY_START_LOC_INDEX);
    OP_CHECK_IF(seqStartIndexStorageShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "QueryStartLoc must be provided"),
                return ge::GRAPH_FAILED);

    // 检查 cacheIndices (必须提供)
    auto cacheIndicesShape = context_->GetOptionalInputShape(INPUT_CACHE_INDICES_INDEX);
    OP_CHECK_IF(cacheIndicesShape == nullptr,
                OP_LOGE(context_->GetNodeName(), "CacheIndices must be provided"),
                return ge::GRAPH_FAILED);

    // 注意：seqStartIndexShape_ 已在 GetShapeAttrsInfo 中处理，这里只做验证
    if (seqStartIndexStorageShape != nullptr) {
        // 如果提供了 queryStartLoc，检查维度
        auto seqStartIndexShape = seqStartIndexStorageShape->GetStorageShape();
        uint64_t seqStartIndexDimNum = seqStartIndexShape.GetDimNum();
        OP_CHECK_IF(seqStartIndexDimNum != SEQ_START_INDEX_DIM_NUM,
                    OP_LOGE(context_->GetNodeName(), "SeqStartIndex dim must be 1, but got: %lu", seqStartIndexDimNum),
                    return ge::GRAPH_FAILED);

        uint64_t seqStartIndexDim0 = seqStartIndexShape.GetDim(DIM_0);
        OP_CHECK_IF(seqStartIndexDim0 != (batch_ + 1),
                    OP_LOGE(context_->GetNodeName(), "SeqStartIndex dim[0] must equal to batch+1=%u, but got: %lu",
                            batch_ + 1, seqStartIndexDim0),
                    return ge::GRAPH_FAILED);
    }

    // 检查batch范围
    OP_CHECK_IF(!(batch_ >= BATCH_MIN && batch_ <= BATCH_MAX),
                OP_LOGE(context_->GetNodeName(), "batch must in [%lu, %lu], but got: %u",
                        BATCH_MIN, BATCH_MAX, batch_),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 检查输入参数
ge::graphStatus CausalConv1dFnTiling::CheckInputParams()
{
    if (CheckInputDtype() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    if (CheckInputDim() != ge::GRAPH_SUCCESS) {
        return ge::GRAPH_FAILED;
    }
    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::CheckOutputParams()
{
    auto outputYDesc = context_->GetOutputDesc(OUTPUT_Y_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputYDesc);
    auto outputYType = outputYDesc->GetDataType();
    OP_CHECK_IF(xType_ != outputYType,
                OP_LOGE(context_->GetNodeName(), "Output Y dtype must equal to X dtype. X dtype: %s, Y dtype: %s",
                        Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(outputYType).c_str()),
                return ge::GRAPH_FAILED);

    auto outputCacheStatesDesc = context_->GetOutputDesc(OUTPUT_CACHE_STATES_INDEX);
    OP_CHECK_NULL_WITH_CONTEXT(context_, outputCacheStatesDesc);
    auto outputCacheStatesType = outputCacheStatesDesc->GetDataType();
    OP_CHECK_IF(xType_ != outputCacheStatesType,
                OP_LOGE(context_->GetNodeName(),
                        "Output CacheStates dtype must equal to X dtype. X dtype: %s, CacheStates dtype: %s",
                        Ops::Base::ToString(xType_).c_str(), Ops::Base::ToString(outputCacheStatesType).c_str()),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::GetShapeAttrsInfo()
{
    OP_CHECK_IF(context_ == nullptr, OP_LOGE("CausalConv1dFn", "context is null"), return ge::GRAPH_FAILED);

    // 获取输入shape
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_X_INDEX));
    xShape_ = context_->GetInputShape(INPUT_X_INDEX)->GetOriginShape();
    cuSeqLen_ = xShape_.GetDim(DIM_0);
    dim_ = xShape_.GetDim(DIM_1);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_WEIGHT_INDEX));
    weightShape_ = context_->GetInputShape(INPUT_WEIGHT_INDEX)->GetOriginShape();
    kernelWidth_ = weightShape_.GetDim(DIM_0);

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_CACHE_STATES_INDEX));
    cacheStatesShape_ = context_->GetInputShape(INPUT_CACHE_STATES_INDEX)->GetOriginShape();

    // 获取 queryStartLoc (OPTIONAL)
    auto seqStartIndexStorageShape = context_->GetOptionalInputShape(INPUT_QUERY_START_LOC_INDEX);
    if (seqStartIndexStorageShape != nullptr) {
        seqStartIndexShape_ = seqStartIndexStorageShape->GetOriginShape();
        batch_ = seqStartIndexShape_.GetDim(DIM_0) - 1;
    } else {
        // 没有提供 queryStartLoc，默认 batch = 1，处理全部序列
        batch_ = 1;
        // 创建一个默认的 gert::Shape，表示没有分批的情况
        seqStartIndexShape_ = gert::Shape({0, cuSeqLen_});
    }

    // 获取输入数据类型
    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_X_INDEX));
    xType_ = context_->GetInputDesc(INPUT_X_INDEX)->GetDataType();

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputDesc(INPUT_WEIGHT_INDEX));
    weightType_ = context_->GetInputDesc(INPUT_WEIGHT_INDEX)->GetDataType();

    xDtypeSize_ = GetSizeByDataType(xType_);
    OP_CHECK_IF(xDtypeSize_ == 0,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dFn get X dtype[%s] size is 0.",
                        Ops::Base::ToString(xType_).c_str()),
                return ge::GRAPH_FAILED);

    // 读取 padSlotId attribute
    padSlotId_ = -1;
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(ATTR_PAD_SLOT_ID_INDEX) != nullptr) {
        padSlotId_ = *(context_->GetAttrs()->GetInt(ATTR_PAD_SLOT_ID_INDEX));
    }

    residualConnection_ = 0;
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(ATTR_RESIDUAL_CONNECTION_INDEX) != nullptr) {
        residualConnection_ = *(context_->GetAttrs()->GetInt(ATTR_RESIDUAL_CONNECTION_INDEX));
    }

    // 初始化有效 batch 范围（默认为全部 batch）
    validBatchStart_ = 0;
    validBatchCount_ = batch_;
    validSeqStart_ = 0;
    validSeqLen_ = cuSeqLen_;

    // 如果有 padSlotId，需要读取 cacheIndices 来确定有效 batch 范围
    if (padSlotId_ >= 0) {
        // 尝试读取 cacheIndices 数据 (OPTIONAL)
        const gert::Tensor* cacheIndicesTensor = context_->GetOptionalInputTensor(INPUT_CACHE_INDICES_INDEX);
        if (cacheIndicesTensor != nullptr && cacheIndicesTensor->GetData<int32_t>() != nullptr) {
            // 获取 cacheIndices tensor (batch 个 int32 元素)
            const int32_t* cacheIndices = cacheIndicesTensor->GetData<int32_t>();

            // 从前往后找第一个不等于 padSlotId 的位置
            uint64_t validStart = batch_;  // 默认全是padding
            for (uint64_t i = 0; i < batch_; i++) {
                if (static_cast<int64_t>(cacheIndices[i]) != padSlotId_) {
                    validStart = i;
                    break;
                }
            }

            // 从后往前找最后一个不等于 padSlotId 的位置
            uint64_t validEnd = 0;
            for (int64_t i = static_cast<int64_t>(batch_) - 1; i >= 0; i--) {
                if (static_cast<int64_t>(cacheIndices[i]) != padSlotId_) {
                    validEnd = static_cast<uint64_t>(i);
                    break;
                }
            }

            // 计算有效 batch 数量
            if (validStart <= validEnd && validStart < batch_) {
                validBatchStart_ = validStart;
                validBatchCount_ = validEnd - validStart + 1;

                // 读取 queryStartLoc 计算有效序列范围 (OPTIONAL)
                const gert::Tensor* queryStartLocTensor = context_->GetOptionalInputTensor(INPUT_QUERY_START_LOC_INDEX);
                if (queryStartLocTensor != nullptr && queryStartLocTensor->GetData<int32_t>() != nullptr) {
                    const int32_t* queryStartLoc = queryStartLocTensor->GetData<int32_t>();
                    validSeqStart_ = static_cast<uint64_t>(queryStartLoc[validBatchStart_]);
                    uint64_t validSeqEnd = static_cast<uint64_t>(queryStartLoc[validEnd + 1]);
                    validSeqLen_ = validSeqEnd - validSeqStart_;
                }
            } else {
                // 所有 batch 都是 padding，设置为 0
                validBatchCount_ = 0;
                validSeqLen_ = 0;
            }
        }
    }

    // 检查输入和输出参数
    OP_CHECK_IF(CheckInputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dFn CheckInputParams FAILED."),
                return ge::GRAPH_FAILED);
    OP_CHECK_IF(CheckOutputParams() != ge::GRAPH_SUCCESS,
                OP_LOGE(context_->GetNodeName(), "CausalConv1dFn CheckOutputParams FAILED."),
                return ge::GRAPH_FAILED);

    return ge::GRAPH_SUCCESS;
}

// 辅助函数：计算切cu_seq_len时的核间切分信息（均分多尾核策略）
CausalConv1dFnTiling::CuSeqLenSplitInfo CausalConv1dFnTiling::CalculateCuSeqLenSplitInfo(
    uint64_t cuSeqLen, uint64_t bsOverlap, uint64_t coreNum) const
{
    CuSeqLenSplitInfo info;

    // 均分策略：将带重叠的总长度均分到所有核
    // effectiveTotal = cuSeqLen + (coreNum - 1) * overlap
    info.effectiveTotal = cuSeqLen + (coreNum - 1) * bsOverlap;

    // 向下取整的基础长度
    info.baseLen = info.effectiveTotal / coreNum;

    // 余数（需要多分配的块数）
    info.remainder = info.effectiveTotal % coreNum;

    // 前 remainder 个核是大核，后面是小核
    if (info.remainder > 0) {
        info.blockFactor = info.baseLen + 1;      // 大核载入长度
        info.blockTailFactor = info.baseLen;      // 小核载入长度
    } else {
        // 所有核均匀分配
        info.blockFactor = info.baseLen;
        info.blockTailFactor = info.baseLen;
    }
    info.realCoreNum = coreNum;               // 所有核都使用

    return info;
}


// 计算二维切分时的tiling（支持不均匀切分 + dim循环）
ge::graphStatus CausalConv1dFnTiling::Calculate2DTiling()
{
    // 核内UB分配策略：
    // 1. 优先保证BS方向能装下整核的数据（bsBlockFactor）
    // 2. dim方向保证最小128（256 bytes对齐），尽量扩大但不超过核间分配的dim
    // 3. dim方向可能需要循环加载

    uint64_t bsOverlap = kernelWidth_ - 1;

    // 固定UB使用
    uint64_t startLocInQueueSize = (batch_ + 1) * sizeof(int32_t);
    uint64_t indicesInQueueSize = batch_ * sizeof(int32_t);
    uint64_t hasInitialInQueueSize = batch_ * sizeof(int32_t);
    uint64_t fixedUbSize = startLocInQueueSize + indicesInQueueSize + hasInitialInQueueSize;

    // ===== 计算整核（大核）的UB参数 =====
    uint64_t coreDim = dimBlockFactor_;      // 核间分配的dim大小（大核）
    uint64_t coreBS = bsBlockFactor_;        // 核间分配的BS大小（大核）

    // weight和cache每个dim元素的系数
    uint64_t weightCacheCoeffPerDim = (kernelWidth_ + kernelWidth_ - 1) * xDtypeSize_;

    // x每个dim元素的系数（双buffer，满BS）
    uint64_t xCoeffPerDimFullBS = coreBS * xDtypeSize_ * DOUBLE_BUFFER_NUM;

    // 总系数
    uint64_t totalCoeffPerDim = weightCacheCoeffPerDim + xCoeffPerDimFullBS;

    // 计算可用UB和最大ubDim
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - fixedUbSize;
    int64_t maxUbDim = availableUbSize / totalCoeffPerDim;

    // 对齐到DIM_ALIGN_ELEMENTS (128)
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;

    if (maxUbDim >= static_cast<int64_t>(DIM_ALIGN_ELEMENTS)) {
        // 能装下满BS，尽量扩大dim
        ubFactorBS_ = coreBS;
        ubFactorDim_ = std::min(static_cast<uint64_t>(maxUbDim), coreDim);

        // 确保ubFactorDim_对齐到DIM_ALIGN_ELEMENTS
        ubFactorDim_ = (ubFactorDim_ / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;
        if (ubFactorDim_ == 0) {
            ubFactorDim_ = DIM_ALIGN_ELEMENTS;
        }
    } else {
        // 不能装下满BS，使用最小dim并减少BS
        ubFactorDim_ = DIM_ALIGN_ELEMENTS;

        // weight和cache占用空间（使用最小dim）
        uint64_t weightCacheSize = weightCacheCoeffPerDim * ubFactorDim_;
        int64_t availableForX = availableUbSize - weightCacheSize;

        // x每个BS的大小（双buffer）
        uint64_t xSizePerBS = ubFactorDim_ * xDtypeSize_ * DOUBLE_BUFFER_NUM;

        // 计算能装下多少BS
        int64_t maxBS = availableForX / xSizePerBS;
        ubFactorBS_ = std::max(maxBS, static_cast<int64_t>(1));
        ubFactorBS_ = std::min(ubFactorBS_, coreBS);

        if (ubFactorBS_ == 0) {
            OP_LOGE(context_->GetNodeName(), "UB size is not enough for tiling");
            return ge::GRAPH_FAILED;
        }
    }

    // 计算dim方向的循环次数
    if (coreDim <= ubFactorDim_) {
        loopNumDim_ = 1;
        ubTailFactorDim_ = ubFactorDim_;
    } else {
        loopNumDim_ = (coreDim + ubFactorDim_ - 1) / ubFactorDim_;
        ubTailFactorDim_ = coreDim - (loopNumDim_ - 1) * ubFactorDim_;
    }

    // 计算BS方向的循环次数（考虑overlap）
    uint64_t coreLoopsNeeded;
    if (bsBlockFactor_ <= ubFactorBS_) {
        coreLoopsNeeded = 1;
    } else {
        uint64_t remaining = bsBlockFactor_ - ubFactorBS_;
        uint64_t subsequentLoops = Ops::Base::CeilDiv(remaining, ubFactorBS_ - bsOverlap);
        coreLoopsNeeded = 1 + subsequentLoops;
    }
    loopNumBS_ = coreLoopsNeeded;

    // 计算整核最后一次循环载入大小
    uint64_t coreLastLoopInput = bsBlockFactor_ - (coreLoopsNeeded - 1) * (ubFactorBS_ - bsOverlap);
    ubTailFactorBS_ = std::min(coreLastLoopInput, ubFactorBS_);

    // ===== 计算尾核（双重小核）的UB参数 =====
    // 使用相同的逻辑，但基于尾核的dim和BS大小
    uint64_t tailCoreDim = (dimRemainderCores_ > 0) ? dimBlockTailFactor_ : dimBlockFactor_;
    uint64_t tailCoreBS = bsBlockTailFactor_;

    // 重新计算尾核的UB参数
    uint64_t tailXCoeffPerDimFullBS = tailCoreBS * xDtypeSize_ * DOUBLE_BUFFER_NUM;
    uint64_t tailTotalCoeffPerDim = weightCacheCoeffPerDim + tailXCoeffPerDimFullBS;
    int64_t tailMaxUbDim = availableUbSize / tailTotalCoeffPerDim;
    tailMaxUbDim = (tailMaxUbDim / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;

    if (tailMaxUbDim >= static_cast<int64_t>(DIM_ALIGN_ELEMENTS)) {
        tailBlockubFactorBS_ = tailCoreBS;
        tailBlockubFactorDim_ = std::min(static_cast<uint64_t>(tailMaxUbDim), tailCoreDim);
        tailBlockubFactorDim_ = (tailBlockubFactorDim_ / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;
        if (tailBlockubFactorDim_ == 0) {
            tailBlockubFactorDim_ = DIM_ALIGN_ELEMENTS;
        }
    } else {
        tailBlockubFactorDim_ = DIM_ALIGN_ELEMENTS;
        uint64_t weightCacheSize = weightCacheCoeffPerDim * tailBlockubFactorDim_;
        int64_t availableForX = availableUbSize - weightCacheSize;
        uint64_t xSizePerBS = tailBlockubFactorDim_ * xDtypeSize_ * DOUBLE_BUFFER_NUM;
        int64_t maxBS = availableForX / xSizePerBS;
        tailBlockubFactorBS_ = std::max(maxBS, static_cast<int64_t>(1));
        tailBlockubFactorBS_ = std::min(tailBlockubFactorBS_, tailCoreBS);

        if (tailBlockubFactorBS_ == 0) {
            OP_LOGE(context_->GetNodeName(), "UB size is not enough for tail block tiling");
            return ge::GRAPH_FAILED;
        }
    }

    // 计算尾核dim方向的循环次数
    if (tailCoreDim <= tailBlockubFactorDim_) {
        tailBlockloopNumDim_ = 1;
        tailBlockubTailFactorDim_ = tailBlockubFactorDim_;
    } else {
        tailBlockloopNumDim_ = (tailCoreDim + tailBlockubFactorDim_ - 1) / tailBlockubFactorDim_;
        tailBlockubTailFactorDim_ = tailCoreDim - (tailBlockloopNumDim_ - 1) * tailBlockubFactorDim_;
    }

    // 计算尾核BS方向的循环次数
    uint64_t tailLoopsNeeded;
    if (bsBlockTailFactor_ <= tailBlockubFactorBS_) {
        tailLoopsNeeded = 1;
    } else {
        uint64_t remaining = bsBlockTailFactor_ - tailBlockubFactorBS_;
        uint64_t subsequentLoops = Ops::Base::CeilDiv(remaining, tailBlockubFactorBS_ - bsOverlap);
        tailLoopsNeeded = 1 + subsequentLoops;
    }
    tailBlockloopNumBS_ = tailLoopsNeeded;

    uint64_t tailLastLoopInput = bsBlockTailFactor_ - (tailLoopsNeeded - 1) * (tailBlockubFactorBS_ - bsOverlap);
    tailBlockubTailFactorBS_ = std::min(tailLastLoopInput, tailBlockubFactorBS_);

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::DoOpTiling()
{
    // 二维切分策略（不均匀切分）：先切dim（以128为粒度，允许不均匀分配），再切BS（考虑overlap）
    // 目标：最大化核利用率，优先切dim方向

    uint64_t bsOverlap = kernelWidth_ - 1;
    constexpr uint64_t DIM_GRANULARITY = DIM_ALIGN_ELEMENTS;  // 128

    // 步骤1：计算dim方向可切的份数（N = dim / 128）
    uint64_t N = dim_ / DIM_GRANULARITY;
    if (N == 0) {
        OP_LOGE(context_->GetNodeName(), "dim %lu is smaller than DIM_GRANULARITY %lu",
                dim_, DIM_GRANULARITY);
        return ge::GRAPH_FAILED;
    }

    // 步骤2：贪心搜索最优(dimCoreNum, bsCoreNum)组合
    // 优先尝试dim切分多的方案（从N向下遍历所有可能值，允许不均匀切分）

    // 初始化为 dc=1 的情况（所有核给BS方向）
    uint64_t bestDimCores = 1;
    uint64_t bestBSCores = 1;
    uint64_t bestUsed = 1;
    CuSeqLenSplitInfo bestBSSplitInfo;

    // 从大到小遍历 [1, N] 的所有值（允许不均匀切分）
    for (uint64_t dc = N; dc >= 1; --dc) {
        // 计算每核分配的128-块数
        uint64_t base = N / dc;
        if (base == 0) continue;  // 跳过（每个核至少要1个128-块）

        // 计算该dimCores下能分配的最大BS核数
        uint64_t maxAllowedBSByCore = totalCoreNum_ / dc;
        if (maxAllowedBSByCore == 0) continue;  // 跳过（dim切太多，没有剩余核给BS）

        // BS方向的约束：n <= validSeqLen - overlap（保证每个核至少输出1个元素）
        uint64_t maxAllowedBSBySeqLen = (validSeqLen_ > bsOverlap) ? (validSeqLen_ - bsOverlap) : 1;
        uint64_t maxAllowedBS = std::min(maxAllowedBSByCore, maxAllowedBSBySeqLen);

        // 考虑因果重叠，计算BS方向实际能用的核数
        auto splitInfo = CalculateCuSeqLenSplitInfo(validSeqLen_, bsOverlap, maxAllowedBS);
        uint64_t actualBS = splitInfo.realCoreNum;

        // 总使用核数
        uint64_t usedCores = dc * actualBS;

        // 更新最优解（优先核数多，核数相同优先dim切分多）
        if (usedCores > bestUsed || (usedCores == bestUsed && dc > bestDimCores)) {
            bestDimCores = dc;
            bestBSCores = actualBS;
            bestUsed = usedCores;
            bestBSSplitInfo = splitInfo;  // 保存BS方向的切分信息
        }

        // 如果已经完美利用所有核，提前退出
        if (bestUsed == totalCoreNum_) {
            break;
        }
    }

    // 步骤3：计算dim方向不均匀分配参数
    uint64_t base = N / bestDimCores;          // 每个小核分到的128-块数
    uint64_t remainder = N % bestDimCores;     // 需要多分配的块数（大核数量）

    dimCoreNum_ = bestDimCores;
    dimRemainderCores_ = remainder;

    if (remainder > 0) {
        // 有大核：前remainder个核是大核
        dimBlockFactor_ = (base + 1) * DIM_GRANULARITY;     // 大核的dim大小
        dimBlockTailFactor_ = base * DIM_GRANULARITY;       // 小核的dim大小
    } else {
        // 均匀分配：所有核大小相同
        dimBlockFactor_ = base * DIM_GRANULARITY;
        dimBlockTailFactor_ = base * DIM_GRANULARITY;
    }

    // 步骤4：保存BS方向分配结果
    bsCoreNum_ = bestBSCores;
    bsRemainderCores_ = bestBSSplitInfo.remainder;        // BS方向大核数量
    bsBlockFactor_ = bestBSSplitInfo.blockFactor;         // BS方向大核长度
    bsBlockTailFactor_ = bestBSSplitInfo.blockTailFactor; // BS方向小核长度

    // 步骤5：核数信息
    realCoreNum_ = bestUsed;

    // 步骤6：计算二维切分下的UB参数
    return Calculate2DTiling();
}

uint64_t CausalConv1dFnTiling::GetTilingKey() const
{
    // 根据数据类型返回不同的 tiling key
    if (xType_ == ge::DataType::DT_BF16) {
        return TILING_KEY_FN_BF16;
    } else if (xType_ == ge::DataType::DT_FLOAT16) {
        return TILING_KEY_FN_FP16;
    }
}

ge::graphStatus CausalConv1dFnTiling::GetWorkspaceSize()
{
    // 基础系统 workspace 大小
    uint64_t baseWorkspaceSize = SYS_WORKSPACE_SIZE;

    // 额外申请一个 seq 的空间，大小为 dim * realCoreNum * byte
    uint64_t seqWorkspaceSize = (kernelWidth_ - 1) * dim_ * realCoreNum_ * xDtypeSize_;

    // 总 workspace 大小
    workspaceSize_ = baseWorkspaceSize + seqWorkspaceSize;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::PostTiling()
{
    auto workspaces = context_->GetWorkspaceSizes(1);
    OP_CHECK_NULL_WITH_CONTEXT(context_, workspaces);
    workspaces[0] = workspaceSize_;

    // Set block dimension (number of cores to use)
    context_->SetBlockDim(realCoreNum_);

    // Clear tiling data to avoid uninitialized padding bytes
    errno_t ret = memset_s(&tilingData_, sizeof(tilingData_), 0, sizeof(tilingData_));
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memset_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }

    // Populate tiling data
    tilingData_.loopNumBS = loopNumBS_;
    tilingData_.loopNumDim = loopNumDim_;
    tilingData_.ubFactorBS = ubFactorBS_;
    tilingData_.ubTailFactorBS = ubTailFactorBS_;
    tilingData_.ubFactorDim = ubFactorDim_;
    tilingData_.ubTailFactorDim = ubTailFactorDim_;
    tilingData_.tailBlockloopNumBS = tailBlockloopNumBS_;
    tilingData_.tailBlockloopNumDim = tailBlockloopNumDim_;
    tilingData_.tailBlockubFactorBS = tailBlockubFactorBS_;
    tilingData_.tailBlockubTailFactorBS = tailBlockubTailFactorBS_;
    tilingData_.tailBlockubFactorDim = tailBlockubFactorDim_;
    tilingData_.tailBlockubTailFactorDim = tailBlockubTailFactorDim_;

    // dim方向核间切分信息
    tilingData_.dimCoreNum = dimCoreNum_;
    tilingData_.dimRemainderCores = dimRemainderCores_;
    tilingData_.dimBlockFactor = dimBlockFactor_;
    tilingData_.dimBlockTailFactor = dimBlockTailFactor_;

    // BS方向核间切分信息
    tilingData_.bsCoreNum = bsCoreNum_;
    tilingData_.bsRemainderCores = bsRemainderCores_;
    tilingData_.bsBlockFactor = bsBlockFactor_;
    tilingData_.bsBlockTailFactor = bsBlockTailFactor_;

    // 核数信息
    tilingData_.realCoreNum = realCoreNum_;

    // 其他参数
    tilingData_.kernelWidth = kernelWidth_;
    tilingData_.cuSeqLen = cuSeqLen_;
    tilingData_.dim = dim_;
    tilingData_.batch = batch_;
    tilingData_.validBatchStart = validBatchStart_;
    tilingData_.validBatchCount = validBatchCount_;
    tilingData_.validSeqStart = validSeqStart_;
    tilingData_.validSeqLen = validSeqLen_;
    tilingData_.xStride = dim_;
    tilingData_.cacheStride = dim_;
    tilingData_.residualConnection = residualConnection_;

    // Save tiling data to buffer
    auto tilingDataSize = sizeof(CausalConv1dFnTilingData);
    ret = memcpy_s(context_->GetRawTilingData()->GetData(),
                    context_->GetRawTilingData()->GetCapacity(),
                    reinterpret_cast<void *>(&tilingData_), tilingDataSize);
    if (ret != EOK) {
        OP_LOGE(context_->GetNodeName(), "memcpy_s failed, ret=%d", ret);
        return ge::GRAPH_FAILED;
    }
    context_->GetRawTilingData()->SetDataSize(tilingDataSize);

    return ge::GRAPH_SUCCESS;
}

void CausalConv1dFnTiling::DumpTilingInfo()
{
    std::ostringstream info;
    info << "cuSeqLen: " << cuSeqLen_ << std::endl;
    info << "dim: " << dim_ << std::endl;
    info << "kernelWidth: " << kernelWidth_ << std::endl;
    info << "batch: " << batch_ << std::endl;
    info << "padSlotId: " << padSlotId_ << std::endl;
    info << "validBatchStart: " << validBatchStart_ << std::endl;
    info << "validBatchCount: " << validBatchCount_ << std::endl;
    info << "validSeqStart: " << validSeqStart_ << std::endl;
    info << "validSeqLen: " << validSeqLen_ << std::endl;

    // dim方向核间切分信息
    info << "dimCoreNum: " << dimCoreNum_ << std::endl;
    info << "dimRemainderCores: " << dimRemainderCores_ << std::endl;
    info << "dimBlockFactor: " << dimBlockFactor_ << std::endl;
    info << "dimBlockTailFactor: " << dimBlockTailFactor_ << std::endl;

    // BS方向核间切分信息
    info << "bsCoreNum: " << bsCoreNum_ << std::endl;
    info << "bsRemainderCores: " << bsRemainderCores_ << std::endl;
    info << "bsBlockFactor: " << bsBlockFactor_ << std::endl;
    info << "bsBlockTailFactor: " << bsBlockTailFactor_ << std::endl;

    // 核数信息
    info << "realCoreNum: " << realCoreNum_ << std::endl;

    // 核内切分参数
    info << "loopNumBS: " << loopNumBS_ << std::endl;
    info << "loopNumDim: " << loopNumDim_ << std::endl;
    info << "ubFactorBS: " << ubFactorBS_ << std::endl;
    info << "ubTailFactorBS: " << ubTailFactorBS_ << std::endl;
    info << "ubFactorDim: " << ubFactorDim_ << std::endl;
    info << "ubTailFactorDim: " << ubTailFactorDim_ << std::endl;
    info << "tailBlockloopNumBS: " << tailBlockloopNumBS_ << std::endl;
    info << "tailBlockloopNumDim: " << tailBlockloopNumDim_ << std::endl;
    info << "tailBlockubFactorBS: " << tailBlockubFactorBS_ << std::endl;
    info << "tailBlockubTailFactorBS: " << tailBlockubTailFactorBS_ << std::endl;
    info << "tailBlockubFactorDim: " << tailBlockubFactorDim_ << std::endl;
    info << "tailBlockubTailFactorDim: " << tailBlockubTailFactorDim_ << std::endl;
    info << "residualConnection: " << residualConnection_ << std::endl;

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

// REGISTER_OPS_TILING_TEMPLATE(CausalConv1dFn, CausalConv1dFnTiling, 1);
} // namespace optiling
