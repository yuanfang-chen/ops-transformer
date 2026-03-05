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

constexpr uint64_t ALIGN_BYTES = 256;
constexpr uint64_t DIM_ALIGN_ELEMENTS = 128;  // 256 bytes / 2 bytes per element (fp16/bf16)
constexpr uint64_t SYSTEM_RESERVED_UB_SIZE = 8 * 1024;  // 8 KB system reserved UB space
constexpr uint64_t MIN_CORE_NUM_FOR_DIM_SPLIT = 32;
constexpr uint64_t MIN_DIM_PER_CORE = 256;
constexpr uint64_t DOUBLE_BUFFER_NUM = 2;
constexpr uint64_t TILING_KEY_VALUE = 10000UL;
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
    kernelWidth_ = static_cast<uint32_t>(weightShape_.GetDim(DIM_0));

    OP_CHECK_NULL_WITH_CONTEXT(context_, context_->GetInputShape(INPUT_CACHE_STATES_INDEX));
    cacheStatesShape_ = context_->GetInputShape(INPUT_CACHE_STATES_INDEX)->GetOriginShape();

    // 获取 queryStartLoc (OPTIONAL)
    auto seqStartIndexStorageShape = context_->GetOptionalInputShape(INPUT_QUERY_START_LOC_INDEX);
    if (seqStartIndexStorageShape != nullptr) {
        seqStartIndexShape_ = seqStartIndexStorageShape->GetOriginShape();
        batch_ = static_cast<uint32_t>(seqStartIndexShape_.GetDim(DIM_0) - 1);
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
    if (context_->GetAttrs() != nullptr && context_->GetAttrs()->GetInt(0) != nullptr) {
        padSlotId_ = *(context_->GetAttrs()->GetInt(0));
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
            uint32_t validStart = batch_;  // 默认全是padding
            for (uint32_t i = 0; i < batch_; i++) {
                if (static_cast<int64_t>(cacheIndices[i]) != padSlotId_) {
                    validStart = i;
                    break;
                }
            }

            // 从后往前找最后一个不等于 padSlotId 的位置
            uint32_t validEnd = 0;
            for (int32_t i = static_cast<int32_t>(batch_) - 1; i >= 0; i--) {
                if (static_cast<int64_t>(cacheIndices[i]) != padSlotId_) {
                    validEnd = static_cast<uint32_t>(i);
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

// 辅助函数：计算切cu_seq_len时的核间切分信息（考虑因果卷积重叠）
CausalConv1dFnTiling::CuSeqLenSplitInfo CausalConv1dFnTiling::CalculateCuSeqLenSplitInfo(
    uint64_t cuSeqLen, uint64_t bsOverlap) const
{
    CuSeqLenSplitInfo info;

    info.effectiveTotal = cuSeqLen + (totalCoreNum_ - 1) * bsOverlap;
    info.baseLen = info.effectiveTotal / totalCoreNum_;
    info.remainder = info.effectiveTotal % totalCoreNum_;

    // blockFactor: 整核的处理长度（前 remainder 个核）
    info.blockFactor = info.baseLen + (info.remainder > 0 ? 1 : 0);

    // 通过累积输出计算实际核数和尾核长度
    uint64_t accumulated_output = 0;
    info.realCoreNum = 0;
    info.blockTailFactor = 0;

    for (uint64_t core_id = 0; core_id < totalCoreNum_; ++core_id) {
        uint64_t core_load_length = info.baseLen + (core_id < info.remainder ? 1 : 0);

        // 这个核能产生的实际输出（去除与前面核的重叠）
        uint64_t core_output;
        if (core_id == 0) {
            core_output = core_load_length;
        } else {
            // 后续核：载入量 - 重叠部分 = 输出
            core_output = core_load_length - bsOverlap;
        }

        accumulated_output += core_output;

        if (accumulated_output >= cuSeqLen) {
            // 这个核已经能覆盖所有数据
            info.realCoreNum = core_id + 1;

            // 计算尾核实际需要的载入长度
            uint64_t excess = accumulated_output - cuSeqLen;
            info.blockTailFactor = core_load_length - excess;
            break;
        }
    }

    // 如果循环结束还没break，说明所有核都需要
    if (info.realCoreNum == 0) {
        info.realCoreNum = totalCoreNum_;
        info.blockTailFactor = info.baseLen + (totalCoreNum_ - 1 < info.remainder ? 1 : 0);
    }

    return info;
}

// 计算切cu_seq_len时的tiling
ge::graphStatus CausalConv1dFnTiling::CalculateCuSeqLenTiling()
{
    blockIndex_ = 0; // 切cu_seq_len
    uint64_t bsOverlap = kernelWidth_ - 1;

    // 步骤1: 核间切分策略（均分带重叠的总长度）
    //
    // 核心思想：
    // 1. 计算带重叠的总长度：effective_total = total_data + (num_cores - 1) * overlap
    // 2. 均分这个总长度到所有核
    // 3. 每个核的起始位置 = 前一个核的结束位置 - overlap
    //
    // 示例（total_data=13, overlap=2, num_cores=3）：
    //   effective_total = 13 + (3-1)*2 = 17
    //   base_len = 17/3 = 5, remainder = 17%3 = 2
    //   核0: start=0, length=6(5+1), end=6
    //   核1: start=4(6-2), length=6(5+1), end=10
    //   核2: start=8(10-2), length=5, end=13(强制)
    //
    // 注意：使用 validBatchCount_ 而不是 batch_ 进行核间切分

    // 检查是否已经在 DoOpTiling 中计算过
    CuSeqLenSplitInfo splitInfo;
    if (hasCachedSplitInfo_) {
        // 使用缓存的结果
        splitInfo = cachedSplitInfo_;
    } else {
        // 第一次计算，使用有效序列长度
        splitInfo = CalculateCuSeqLenSplitInfo(validSeqLen_, bsOverlap);
    }

    // 使用计算结果
    blockFactor_ = splitInfo.blockFactor;
    blockTailFactor_ = static_cast<uint32_t>(splitInfo.blockTailFactor);
    realCoreNum_ = splitInfo.realCoreNum;

    // 步骤2: 核内UB切分参数
    // 根据需求文档6节Buffer设计,计算UB使用
    // 固定UB使用（真正不变的辅助tensor）：
    // startLocInQueue: (batch + 1) * sizeof(int32), BUF_NUM=1
    // indicesInQueue: batch * sizeof(int32), BUF_NUM=1
    // hasInitialInQueue: batch * sizeof(int32), BUF_NUM=1
    //
    // 可变UB使用（取决于ubDim）：
    // weightInQueue: kernelWidth * ubDim * xDtypeSize_, BUF_NUM=1
    // cacheQueue: (kernelWidth-1) * ubDim * xDtypeSize_, BUF_NUM=1
    // xQueue(y复用): ubBS * ubDim * xDtypeSize_, BUF_NUM=2

    uint64_t startLocInQueueSize = (batch_ + 1) * sizeof(int32_t);
    uint64_t indicesInQueueSize = batch_ * sizeof(int32_t);
    uint64_t hasInitialInQueueSize = batch_ * sizeof(int32_t);
    uint64_t fixedUbSize = startLocInQueueSize + indicesInQueueSize + hasInitialInQueueSize;

    // 每个核分到的最大BS（含重叠）
    uint64_t coreBS = blockFactor_;

    // weight 和 cache 每个 dim 元素的系数
    uint64_t weightCacheCoeffPerDim = (kernelWidth_ + kernelWidth_ - 1) * xDtypeSize_;

    // x 每个 dim 元素的系数（双 buffer，满 BS）
    uint64_t xCoeffPerDimFullBS = coreBS * xDtypeSize_ * DOUBLE_BUFFER_NUM;

    // 总系数（每个 dim 元素）
    uint64_t totalCoeffPerDim = weightCacheCoeffPerDim + xCoeffPerDimFullBS;

    // 计算可用 UB 和最大 ubDim
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - fixedUbSize;
    int64_t maxUbDim = availableUbSize / totalCoeffPerDim;

    // 对齐到 DIM_ALIGN_ELEMENTS (256 bytes)
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;

    if (maxUbDim >= DIM_ALIGN_ELEMENTS) {
        // 能装下满 BS，尽量扩大 dim
        ubFactorBS_ = static_cast<uint32_t>(coreBS);
        ubFactorDim_ = static_cast<uint32_t>(std::min(static_cast<uint64_t>(maxUbDim), dim_));

        // 确保 ubFactorDim_ 对齐到 DIM_ALIGN_ELEMENTS
        ubFactorDim_ = (ubFactorDim_ / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;
        if (ubFactorDim_ == 0) {
            ubFactorDim_ = DIM_ALIGN_ELEMENTS;
        }
    } else {
        // 不能装下满 BS，使用最小 dim 并减少 BS
        ubFactorDim_ = DIM_ALIGN_ELEMENTS;

        // weight 和 cache 占用空间（使用最小 dim）
        uint64_t weightCacheSize = weightCacheCoeffPerDim * ubFactorDim_;
        int64_t availableForX = availableUbSize - weightCacheSize;

        // x 每个 BS 的大小（双 buffer）
        uint64_t xSizePerBS = ubFactorDim_ * xDtypeSize_ * DOUBLE_BUFFER_NUM;

        // 计算能装下多少 BS
        int64_t maxBS = availableForX / xSizePerBS;
        ubFactorBS_ = static_cast<uint32_t>(std::max(maxBS, static_cast<int64_t>(1)));
        ubFactorBS_ = std::min(static_cast<uint64_t>(ubFactorBS_), coreBS);

        if (ubFactorBS_ == 0) {
            OP_LOGE(context_->GetNodeName(), "UB size is not enough for tiling");
            return ge::GRAPH_FAILED;
        }
    }

    // 步骤3: 计算整核和尾核的循环次数及最后一次循环载入大小
    //
    // 注意：blockFactor_ 和 blockTailFactor_ 表示每个核的处理长度（包含重叠部分）
    // 这个长度是该核实际需要载入的数据量

    // 计算整核需要多少次循环
    uint64_t coreLoopsNeeded;
    if (blockFactor_ <= ubFactorBS_) {
        coreLoopsNeeded = 1;
    } else {
        uint64_t remaining = blockFactor_ - ubFactorBS_;
        uint64_t subsequentLoops = Ops::Base::CeilDiv(remaining, ubFactorBS_ - bsOverlap);
        coreLoopsNeeded = 1 + subsequentLoops;
    }
    loopNumBS_ = static_cast<uint32_t>(coreLoopsNeeded);

    // 计算整核最后一次循环载入大小
    uint64_t coreLastLoopInput = blockFactor_ - (coreLoopsNeeded - 1) * (ubFactorBS_ - bsOverlap);
    ubTailFactorBS_ = static_cast<uint32_t>(std::min(coreLastLoopInput, static_cast<uint64_t>(ubFactorBS_)));

    // 计算尾核需要多少次循环
    uint64_t tailLoopsNeeded;
    if (blockTailFactor_ <= ubFactorBS_) {
        tailLoopsNeeded = 1;
    } else {
        uint64_t remaining = blockTailFactor_ - ubFactorBS_;
        uint64_t subsequentLoops = Ops::Base::CeilDiv(remaining, ubFactorBS_ - bsOverlap);
        tailLoopsNeeded = 1 + subsequentLoops;
    }
    tailBlockloopNumBS_ = static_cast<uint32_t>(tailLoopsNeeded);

    // 计算尾核最后一次循环载入大小
    uint64_t tailLastLoopInput = blockTailFactor_ - (tailLoopsNeeded - 1) * (ubFactorBS_ - bsOverlap);
    tailBlockubTailFactorBS_ = static_cast<uint32_t>(std::min(tailLastLoopInput, static_cast<uint64_t>(ubFactorBS_)));

    // 步骤4: Dim方向参数
    loopNumDim_ = static_cast<uint32_t>(Ops::Base::CeilDiv(dim_, static_cast<uint64_t>(ubFactorDim_)));
    ubTailFactorDim_ = static_cast<uint32_t>(dim_ - (loopNumDim_ - 1) * ubFactorDim_);

    // 步骤5: 尾核其他参数
    tailBlockubFactorBS_ = ubFactorBS_;
    tailBlockloopNumDim_ = loopNumDim_;
    tailBlockubFactorDim_ = ubFactorDim_;
    tailBlockubTailFactorDim_ = ubTailFactorDim_;

    return ge::GRAPH_SUCCESS;
}

// 计算切dim时的tiling
ge::graphStatus CausalConv1dFnTiling::CalculateDimTiling()
{
    blockIndex_ = 1; // 切dim
    uint64_t bsOverlap = kernelWidth_ - 1;

    // 步骤1: 核间切分 - dim 维度均分到所有核
    // 注意：dim 切分时，核间没有重叠（不像 cu_seq_len 有因果重叠）

    // 计算每个核分到的 dim 大小（向上取整）
    uint64_t dim_per_core = Ops::Base::CeilDiv(dim_, totalCoreNum_);

    // 计算实际需要的核数
    realCoreNum_ = Ops::Base::CeilDiv(dim_, dim_per_core);

    // blockFactor_: 整核处理的 dim 大小
    blockFactor_ = dim_per_core;

    // blockTailFactor_: 尾核处理的 dim 大小
    blockTailFactor_ = dim_ - (realCoreNum_ - 1) * blockFactor_;

    // 步骤2: 核内UB切分参数
    // 根据需求文档6节Buffer设计,计算UB使用
    // 固定UB使用（真正不变的辅助tensor）：
    // startLocInQueue: (batch + 1) * sizeof(int32), BUF_NUM=1
    // indicesInQueue: batch * sizeof(int32), BUF_NUM=1
    // hasInitialInQueue: batch * sizeof(int32), BUF_NUM=1
    //
    // 可变UB使用（取决于ubDim）：
    // weightInQueue: kernelWidth * ubDim * xDtypeSize_, BUF_NUM=1
    // cacheQueue: (kernelWidth-1) * ubDim * xDtypeSize_, BUF_NUM=1
    // xQueue(y复用): ubBS * ubDim * xDtypeSize_, BUF_NUM=2

    uint64_t startLocInQueueSize = (batch_ + 1) * sizeof(int32_t);
    uint64_t indicesInQueueSize = batch_ * sizeof(int32_t);
    uint64_t hasInitialInQueueSize = batch_ * sizeof(int32_t);
    uint64_t fixedUbSize = startLocInQueueSize + indicesInQueueSize + hasInitialInQueueSize;

    // 每个核分到的 dim 大小
    uint64_t coreDim = blockFactor_;
    // 完整的 BS 长度
    uint64_t coreBS = validSeqLen_;

    // weight 和 cache 每个 dim 元素的系数
    uint64_t weightCacheCoeffPerDim = (kernelWidth_ + kernelWidth_ - 1) * xDtypeSize_;

    // x 每个 dim 元素的系数（双 buffer，满 BS）
    uint64_t xCoeffPerDimFullBS = coreBS * xDtypeSize_ * DOUBLE_BUFFER_NUM;

    // 总系数（每个 dim 元素）
    uint64_t totalCoeffPerDim = weightCacheCoeffPerDim + xCoeffPerDimFullBS;

    // 计算可用 UB 和最大 ubDim
    int64_t availableUbSize = static_cast<int64_t>(ubSize_) - fixedUbSize;
    int64_t maxUbDim = availableUbSize / totalCoeffPerDim;

    // 对齐到 DIM_ALIGN_ELEMENTS (256 bytes)
    maxUbDim = (maxUbDim / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;

    if (maxUbDim >= DIM_ALIGN_ELEMENTS) {
        // 能装下满 BS，尽量扩大 dim
        ubFactorBS_ = static_cast<uint32_t>(coreBS);
        ubFactorDim_ = static_cast<uint32_t>(std::min(static_cast<uint64_t>(maxUbDim), coreDim));

        // 确保 ubFactorDim_ 对齐到 DIM_ALIGN_ELEMENTS
        ubFactorDim_ = (ubFactorDim_ / DIM_ALIGN_ELEMENTS) * DIM_ALIGN_ELEMENTS;
        if (ubFactorDim_ == 0) {
            ubFactorDim_ = DIM_ALIGN_ELEMENTS;
        }
    } else {
        // 不能装下满 BS，使用最小 dim 并减少 BS
        ubFactorDim_ = DIM_ALIGN_ELEMENTS;

        // weight 和 cache 占用空间（使用最小 dim）
        uint64_t weightCacheSize = weightCacheCoeffPerDim * ubFactorDim_;
        int64_t availableForX = availableUbSize - weightCacheSize;

        // x 每个 BS 的大小（双 buffer）
        uint64_t xSizePerBS = ubFactorDim_ * xDtypeSize_ * DOUBLE_BUFFER_NUM;

        // 计算能装下多少 BS
        int64_t maxBS = availableForX / xSizePerBS;
        ubFactorBS_ = static_cast<uint32_t>(std::max(maxBS, static_cast<int64_t>(1)));
        ubFactorBS_ = std::min(static_cast<uint64_t>(ubFactorBS_), coreBS);

        if (ubFactorBS_ == 0) {
            OP_LOGE(context_->GetNodeName(), "UB size is not enough for tiling in dim mode");
            return ge::GRAPH_FAILED;
        }

        // 检查 ubFactorBS_ 是否大于重叠长度
        if (ubFactorBS_ <= bsOverlap) {
            OP_LOGE(context_->GetNodeName(),
                    "ubFactorBS_=%u is too small in dim mode, must be > kernelWidth-1=%lu",
                    ubFactorBS_, bsOverlap);
            return ge::GRAPH_FAILED;
        }
    }

    // 步骤3: 计算 BS 方向循环次数（考虑重叠）
    // BS 方向有因果卷积重叠，需要按照 CalculateCuSeqLenTiling 的方式计算

    uint64_t bsLoopsNeeded;
    if (validSeqLen_ <= ubFactorBS_) {
        // 一次循环就能处理完
        bsLoopsNeeded = 1;
        ubTailFactorBS_ = static_cast<uint32_t>(validSeqLen_);
    } else {
        // 计算需要多少次循环
        uint64_t remaining = validSeqLen_ - ubFactorBS_;
        uint64_t subsequentLoops = Ops::Base::CeilDiv(remaining, ubFactorBS_ - bsOverlap);
        bsLoopsNeeded = 1 + subsequentLoops;

        // 计算最后一次循环载入大小
        uint64_t lastLoopInput = validSeqLen_ - (bsLoopsNeeded - 1) * (ubFactorBS_ - bsOverlap);
        ubTailFactorBS_ = static_cast<uint32_t>(std::min(lastLoopInput, static_cast<uint64_t>(ubFactorBS_)));
    }
    loopNumBS_ = static_cast<uint32_t>(bsLoopsNeeded);

    // 步骤4: 计算 Dim 方向循环次数（整核）
    loopNumDim_ = static_cast<uint32_t>(Ops::Base::CeilDiv(blockFactor_, static_cast<uint64_t>(ubFactorDim_)));
    ubTailFactorDim_ = static_cast<uint32_t>(blockFactor_ - (loopNumDim_ - 1) * ubFactorDim_);

    // 步骤5: 计算尾核的 dim 方向循环次数
    tailBlockloopNumDim_ = static_cast<uint32_t>(Ops::Base::CeilDiv(blockTailFactor_, static_cast<uint64_t>(ubFactorDim_)));
    tailBlockubTailFactorDim_ = static_cast<uint32_t>(blockTailFactor_ - (tailBlockloopNumDim_ - 1) * ubFactorDim_);

    // 步骤6: 尾核其他参数
    tailBlockloopNumBS_ = loopNumBS_;
    tailBlockubFactorBS_ = ubFactorBS_;
    tailBlockubTailFactorBS_ = ubTailFactorBS_;
    tailBlockubFactorDim_ = ubFactorDim_;

    return ge::GRAPH_SUCCESS;
}

ge::graphStatus CausalConv1dFnTiling::DoOpTiling()
{
    // 根据需求文档5.1节,决定是切cu_seq_len还是切dim
    // 切dim的条件:
    // 1. 内存容量约束: cu_seq_len * dim * xDtypeSize > ubSize / 2
    // 2. 核数限制: 切cu_seq_len时开核数 < 32
    // 3. 每个核处理的dim长度 > 256B
    //
    // 注意：使用 validSeqLen_ 而不是 cuSeqLen_ 进行判断

    bool shouldSplitDim = false;
    uint64_t bsOverlap = kernelWidth_ - 1;

    // 条件1: 内存容量约束（使用有效序列长度）
    uint64_t totalMemory = validSeqLen_ * dim_ * xDtypeSize_;
    bool memoryCondition = (totalMemory > ubSize_ / 2);

    // 条件2: 核数限制（考虑因果卷积重叠，使用有效序列长度）
    // 计算并缓存切分信息，避免在 CalculateCuSeqLenTiling 中重复计算
    cachedSplitInfo_ = CalculateCuSeqLenSplitInfo(validSeqLen_, bsOverlap);
    hasCachedSplitInfo_ = true;

    uint64_t coreNumForCuSeqLen = cachedSplitInfo_.realCoreNum;
    bool coreNumCondition = (coreNumForCuSeqLen < MIN_CORE_NUM_FOR_DIM_SPLIT);

    // 条件3: 每个核处理的dim长度
    uint64_t dimPerCore = Ops::Base::CeilDiv(dim_, totalCoreNum_);
    bool dimPerCoreCondition = (dimPerCore * xDtypeSize_ > MIN_DIM_PER_CORE);

    shouldSplitDim = memoryCondition && coreNumCondition && dimPerCoreCondition;

    if (shouldSplitDim) {
        return CalculateDimTiling();
    } else {
        return CalculateCuSeqLenTiling();
    }
}

uint64_t CausalConv1dFnTiling::GetTilingKey() const
{
    return TILING_KEY_VALUE;
}

ge::graphStatus CausalConv1dFnTiling::GetWorkspaceSize()
{
    // 基础系统 workspace 大小
    uint64_t baseWorkspaceSize = SYS_WORKSPACE_SIZE;

    // 额外申请一个 seq 的空间，大小为 dim * realCoreNum * byte
    uint64_t seqWorkspaceSize = dim_ * realCoreNum_ * xDtypeSize_;

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
    tilingData_.blockFactor = blockFactor_;
    tilingData_.blockIndex = blockIndex_;
    tilingData_.blockTailFactor = blockTailFactor_;
    tilingData_.tailBlockloopNumBS = tailBlockloopNumBS_;
    tilingData_.tailBlockloopNumDim = tailBlockloopNumDim_;
    tilingData_.tailBlockubFactorBS = tailBlockubFactorBS_;
    tilingData_.tailBlockubTailFactorBS = tailBlockubTailFactorBS_;
    tilingData_.tailBlockubFactorDim = tailBlockubFactorDim_;
    tilingData_.tailBlockubTailFactorDim = tailBlockubTailFactorDim_;
    tilingData_.realCoreNum = static_cast<uint32_t>(realCoreNum_);
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
    info << "blockIndex: " << blockIndex_ << std::endl;
    info << "blockFactor: " << blockFactor_ << std::endl;
    info << "blockTailFactor: " << blockTailFactor_ << std::endl;
    info << "realCoreNum: " << realCoreNum_ << std::endl;
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

    OP_LOGI(context_->GetNodeName(), "%s", info.str().c_str());
}

REGISTER_OPS_TILING_TEMPLATE(CausalConv1dFn, CausalConv1dFnTiling, 1);
} // namespace optiling
